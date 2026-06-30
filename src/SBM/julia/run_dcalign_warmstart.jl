#!/usr/bin/env julia
# DCAlign WARM-START fixed-point probe for the two-model combine pipeline.
#
# Diagnostic (iter-003 Phase-B, §10.x): is the native frame a stable fixed point
# of DCAlign's belief-propagation dynamics, or do those dynamics drive it to the
# worse-than-native frame regardless of where they start?
#
#   * If BP, *initialised at the native frame*, STAYS there  → native is reachable;
#     the random-init production runs simply never landed in its basin (a search /
#     initialisation problem → annealing or native-biased init is the lever).
#   * If BP flows AWAY to the same worse frame the random-init runs found → native
#     is not a fixed point of DCAlign's objective (the objective genuinely prefers
#     the other frame → search-tuning is futile).
#
# This is the trustworthy version of the "evaluate the objective at native vs the
# DCAlign frame" question: rather than reconstruct DCAlign's implicit objective
# offline (it has no closed-form per-alignment prior cost — the prior Λ enters only
# inside the BP messages, and is all-pairs), we let DCAlign's OWN inference be the
# oracle by seeding it at native and watching its dynamics.
#
# It REPLICATES the body of `DCAlign.palign` + `DCAlign.update!` (clone pinned at
# commit cab443ffad133e6e68eff8e50b11e8fc59178dbd) and changes ONE thing: the
# random `initialize_all!` is replaced by a deterministic delta initialisation at
# the native frame. The DCAlign clone is NOT edited — every primitive is reached
# via the `DCAlign.` prefix (all are module-reachable; confirmed for this commit).
# So the clone stays a pristine, pinned, read-only dependency on Mac and Midway
# alike, and the only version-controlled change lives in this repo.
#
#   julia --project=/path/to/DCAlign run_dcalign_warmstart.jl <in_dir> <out_tsv>
#
# <in_dir> must contain (written by SBM.utils.dcalign_warmstart.stage_inputs):
#   meta.json        L, q, maxiter, seed, pcount, lambda_spec, alphabet
#                    (+ optional Δβ, thP, Δt; default to DCAlign's palign defaults)
#   model_J.bin      (q,q,L,L) Float64, column-major (DCAlign coupling-major layout)
#   model_h.bin      (q,L)     Float64, column-major
#   queries.fasta    >id / raw gap-free residue string per query (A..Y, no gaps)
#   init.fasta       >id / length-L frame string (residues + '-' gaps) to warm-start
#                    BP from — the native home-model frame OR the fields-MAP frame,
#                    set by build_dcalign_warmstart.py. One per query id; ids must
#                    match queries.fasta. (Legacy name native.fasta is also accepted.)
#   seed.ins         (lambda_spec="deltan" only) the model-frame seed MSA (a2m)
#
# Output (appended to <out_tsv>, one row per query, flushed per row — same 6-column
# schema as run_dcalign.jl so SBM.utils.dcalign_score.read_alignment_cache reads it
# unchanged):
#   seq_id <TAB> aligned_frame <TAB> dcalign_energy <TAB> converged
#          <TAB> used_decimation <TAB> n_iter
# `aligned_frame` is the length-L frame BP settled on FROM the native start; an
# empty frame + NaN energy records a per-sequence failure (record-and-continue).

using DCAlign
using OffsetArrays
using LinearAlgebra

const TSV_HEADER = "seq_id\taligned_frame\tdcalign_energy\tconverged\tused_decimation\tn_iter"

# DCAlign.palign defaults (clone cab443ff) for the annealing schedule; meta.json
# may override. β starts at 1.0 and is COOLED (β += Δβ every Δt sweeps) until the
# marginals saturate — see update! below.
const DEFAULT_DBETA = 0.05
const DEFAULT_THP = 0.30
const DEFAULT_DT = 10
const DEFAULT_DAMP = 0.0


# ── readers: mirror run_dcalign.jl (keep in sync; pinned clone cab443ff) ──────

"""Minimal reader for the flat meta.json (no JSON dep needed; we control it)."""
function read_meta(path::AbstractString)
    txt = read(path, String)
    meta = Dict{String,Any}()
    for m in eachmatch(r"\"(\w+)\"\s*:\s*(\"[^\"]*\"|[-+0-9.eE]+)", txt)
        k = m.captures[1]
        v = m.captures[2]
        meta[k] = startswith(v, "\"") ? String(strip(v, '"')) : parse(Float64, v)
    end
    return meta
end

"""Read the column-major Float64 binaries into dense (q,q,L,L) / (q,L) arrays."""
function read_model(in_dir::AbstractString, q::Int, L::Int)
    jbytes = read(joinpath(in_dir, "model_J.bin"))
    hbytes = read(joinpath(in_dir, "model_h.bin"))
    length(jbytes) == 8 * q * q * L * L ||
        error("model_J.bin is $(length(jbytes)) bytes, expected $(8*q*q*L*L) for (q,q,L,L)=($q,$q,$L,$L)")
    length(hbytes) == 8 * q * L ||
        error("model_h.bin is $(length(hbytes)) bytes, expected $(8*q*L) for (q,L)=($q,$L)")
    J = Array(reshape(reinterpret(Float64, jbytes), (q, q, L, L)))
    h = Array(reshape(reinterpret(Float64, hbytes), (q, L)))
    return J, h
end

"""Tiny FASTA reader (header, sequence) tuples; tolerates wrapped lines."""
function read_fasta(path::AbstractString)
    recs = Tuple{String,String}[]
    name = ""
    buf = IOBuffer()
    for line in eachline(path)
        if startswith(line, ">")
            if name != ""
                push!(recs, (name, String(take!(buf))))
            end
            name = strip(line[2:end])
        else
            print(buf, strip(line))
        end
    end
    if name != ""
        push!(recs, (name, String(take!(buf))))
    end
    return recs
end

"""Build the insertion prior Λ — same two modes as run_dcalign.jl (spec §10.13).

`"flat"` is geometry-blind (mass on Δn=1) and needs no seed file, so it is the
mode used for the Mac-side mechanical smoke test (the `"deltan"` path reads
seed.ins via GZip, which is broken on macOS — the real probe runs on Midway).
`"deltan"` is `DCAlign.deltan_prior(seed.ins, L)`, the prior the production run used.
"""
function build_lambda(spec::AbstractString, L::Int, in_dir::AbstractString)
    if spec == "flat"
        Λ = OffsetArray(fill(0.0, (L, L, 2)), 1:L, 1:L, 0:1)
        for i in 1:L, j in 1:L
            Λ[i, j, 1] = 1.0
        end
        return Λ
    elseif spec == "deltan"
        seedins = joinpath(in_dir, "seed.ins")
        isfile(seedins) ||
            error("lambda_spec=\"deltan\" needs $seedins (written by stage_inputs)")
        return first(DCAlign.deltan_prior(seedins, L))
    end
    error("lambda_spec must be \"flat\" or \"deltan\", got \"$spec\"")
end


# ── native-frame warm start ───────────────────────────────────────────────────

"""Delta BP marginals encoding a length-`L` native frame (the warm-start state).

Returns a `Vector` of `Marginal`s (one per model column), each a one-hot
`OffsetArray(2, N+2)` indexed `[x, n]`, `x∈{0,1}` (1=match, 0=gap), `n∈0:N+1`,
following DCAlign's state convention (`types.jl`, clone cab443ff):

  * match column → residue at query position `n*` ⇒ `[1, n*] = 1`
  * leading gap  (no match seen yet)             ⇒ `[0, 0]   = 1`
  * trailing gap (no match remaining)            ⇒ `[0, N+1] = 1`
  * interior gap                                 ⇒ `[0, n*]  = 1`  (last matched n)

`N` is the number of query residues; the frame must carry exactly `N` non-gap
columns (its non-gap positions are filled by the query residues in order).
"""
function native_marginals(frame::AbstractString, N::Int)
    L = length(frame)
    isres = [c != '-' for c in frame]
    total = count(isres)
    total == N ||
        error("native frame has $total residues but query length N=$N (frame: $frame)")
    margs = Vector{OffsetArray{Float64,2,Array{Float64,2}}}(undef, L)
    seen = 0
    for i in 1:L
        m = OffsetArray(zeros(2, N + 2), 0:1, 0:N+1)
        if isres[i]
            seen += 1
            m[1, seen] = 1.0
        elseif seen == 0
            m[0, 0] = 1.0            # leading gap
        elseif seen == total
            m[0, N+1] = 1.0          # trailing gap (no residue remains)
        else
            m[0, seen] = 1.0         # interior gap, n = last matched index
        end
        margs[i] = m
    end
    return margs
end


# ── BP from a warm start: replica of DCAlign.update! (clone cab443ff) ─────────

"""Run DCAlign belief propagation from the `native` warm start.

A faithful copy of `DCAlign.update!` (clone cab443ff) with ONE change: the random
`initialize_all!` is followed by overwriting the per-column marginals `P/B/F` with
the native delta state. The annealing schedule (β starts at 1.0 and is cooled by
`Δβ` every `Δt` sweeps, scaling `J,h,Λ`) and the saturation convergence test are
reproduced exactly, so the only difference from a production `palign` call is the
starting point. Returns `(n_iter, flagconv::Symbol)`.
"""
function run_bp_from_native!(allvar, native_margs, maxiter::Int, Δβ::Float64,
                             thP::Float64, Δt::Int)
    DCAlign.initialize_all!(allvar)         # valid normalisation for the scratch buffers
    L = allvar.pbf.L
    for i in 1:L
        allvar.pbf.P[i] .= native_margs[i]
        allvar.pbf.B[i] .= native_margs[i]
        allvar.pbf.F[i] .= native_margs[i]
    end
    β = 1.0
    for it in 1:maxiter
        ΔP, ΔB, ΔF = DCAlign.onesweep!(allvar)
        minp, sat = DCAlign.check_solution(allvar)
        if Δβ > 0.0
            (sat && minp > thP) && return it, :converged
        else
            maximum((ΔP, ΔB, ΔF)) < 1e-4 && return it, :converged
        end
        if it % Δt == 0
            β += Δβ
            allvar.jh.J .= β .* allvar.data.J
            allvar.jh.h .= β .* allvar.data.h
            allvar.alg.Λ .= allvar.data.Λ .^ β
        end
    end
    return maxiter, :unconverged
end


"""Warm-start one query at its native frame; returns the TSV row tuple.

Builds the same `AllVar` `palign` builds (clone cab443ff), warm-starts BP at the
native frame, decodes exactly as run_dcalign.jl's `align_one` (decodeposterior +
decimation fallback), and reports the in-frame Potts energy of the settled frame
under the ORIGINAL (unscaled) `J,h`.
"""
function warmstart_one(header, rawseq, native_frame, J, h, Λ, L::Int, maxiter::Int,
                       pcount::Float64, Δβ::Float64, thP::Float64, Δt::Int)
    seq = Seq(String(header), String(rawseq), :amino)
    jh = DCAlign.Jh(deepcopy(J), deepcopy(h))
    pbf = DCAlign.PBF(jh, seq)
    N = pbf.N
    mi = maxiter
    if pbf.L > N                          # palign's fragment allowance
        mi = max(5000, maxiter)
    end
    alg = DCAlign.Alg(false, mi, DEFAULT_DAMP, Λ, 500, pbf.N, pbf.L, pcount;
                      thP=thP, Δβ=Δβ, μext=0.0, μint=0.0, Δt=Δt)
    data = DCAlign.Data(J, h, deepcopy(alg.Λ))
    allvar = DCAlign.AllVar(pbf, jh, seq, alg, data)

    native_margs = native_marginals(String(native_frame), N)
    niter, conv = run_bp_from_native!(allvar, native_margs, mi, Δβ, thP, Δt)

    P = copy(allvar.pbf.P)
    out = DCAlign.decodeposterior(P, allvar.seq.strseq, thP=allvar.alg.thP)
    used_decimation = false
    if !DCAlign.check_assignment(P, false, length(allvar.seq.strseq))
        _, P = DCAlign.decimate_post(allvar, false)
        out = DCAlign.decodeposterior(P, allvar.seq.strseq, thP=allvar.alg.thP)
        used_decimation = true
    end
    frame = out.seq
    length(frame) == L ||
        error("warm-start frame length $(length(frame)) != L=$L for $header")
    energy = DCAlign.compute_potts_en(J, h, frame, L, :amino)
    return (String(header), frame, energy, conv == :converged, used_decimation, Int(niter))
end


function main()
    length(ARGS) >= 2 || error("usage: run_dcalign_warmstart.jl <in_dir> <out_tsv>")
    in_dir = ARGS[1]
    out_tsv = ARGS[2]

    meta = read_meta(joinpath(in_dir, "meta.json"))
    L = Int(meta["L"])
    q = Int(meta["q"])
    maxiter = Int(meta["maxiter"])
    pcount = Float64(meta["pcount"])
    lambda_spec = String(meta["lambda_spec"])
    Δβ = haskey(meta, "Dbeta") ? Float64(meta["Dbeta"]) : DEFAULT_DBETA
    thP = haskey(meta, "thP") ? Float64(meta["thP"]) : DEFAULT_THP
    Δt = haskey(meta, "Dt") ? Int(meta["Dt"]) : DEFAULT_DT

    J, h = read_model(in_dir, q, L)
    Λ = build_lambda(lambda_spec, L, in_dir)
    queries = read_fasta(joinpath(in_dir, "queries.fasta"))
    # The warm-start init frame (native OR fields-MAP, set by build_dcalign_warmstart.py).
    # Prefer init.fasta; fall back to the legacy native.fasta name.
    init_file = isfile(joinpath(in_dir, "init.fasta")) ? "init.fasta" : "native.fasta"
    natives = Dict(read_fasta(joinpath(in_dir, init_file)))
    for (hdr, _) in queries
        haskey(natives, hdr) || error("$init_file has no frame for query id $hdr")
    end
    println(stderr, "run_dcalign_warmstart: L=$L q=$q maxiter=$maxiter pcount=$pcount " *
                    "lambda=$lambda_spec Δβ=$Δβ thP=$thP Δt=$Δt n_queries=$(length(queries)) " *
                    "threads=$(Threads.nthreads())")

    # Per-query independence + per-row flush mirror run_dcalign.jl (resume contract).
    if Threads.nthreads() > 1
        LinearAlgebra.BLAS.set_num_threads(1)
    end
    need_header = !isfile(out_tsv) || filesize(out_tsv) == 0
    iolock = ReentrantLock()
    open(out_tsv, "a") do io
        if need_header
            println(io, TSV_HEADER)
            flush(io)
        end
        Threads.@threads :dynamic for idx in eachindex(queries)
            header, rawseq = queries[idx]
            row = try
                warmstart_one(header, rawseq, natives[header], J, h, Λ, L, maxiter,
                              pcount, Δβ, thP, Δt)
            catch e
                @warn "warm-start failed; recording empty frame" header exception = (e, catch_backtrace())
                (String(header), "", NaN, false, false, 0)
            end
            sid, frame, energy, converged, used_decimation, niter = row
            lock(iolock) do
                println(io, join((sid, frame, energy, converged, used_decimation, niter), '\t'))
                flush(io)
            end
        end
    end
    println(stderr, "run_dcalign_warmstart: done -> $out_tsv")
end

# Guard so the file can be `include`d for unit testing without running main().
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
