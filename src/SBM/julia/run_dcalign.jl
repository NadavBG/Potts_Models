#!/usr/bin/env julia
# DCAlign alignment driver for the SBM two-model combine pipeline (spec §10.9).
#
# Runs the couplings-aware aligner DCAlign (Muntoni et al. 2020) over a set of
# raw query sequences using a Potts model handed over from Python as raw
# little-endian Float64 binaries. This script lives in *our* repo but is run
# with `--project=<DCAlign clone>` so it resolves DCAlign from the clone's env:
#
#   julia --project=/path/to/DCAlign run_dcalign.jl <in_dir> <out_tsv>
#
# <in_dir> must contain (written by SBM.utils.dcalign_score.align_sequences):
#   meta.json      L, q, maxiter, seed, pcount, lambda_spec, alphabet
#   model_J.bin    (q,q,L,L) Float64, column-major  (DCAlign's coupling-major
#                  layout; gap remapped to index 21 — see ORDER in dcalign_score)
#   model_h.bin    (q,L)     Float64, column-major
#   queries.fasta  one >id / raw-sequence (residues A..Y, no gaps) per query
#   seed.ins       (lambda_spec="deltan" only) the model-frame seed MSA as an
#                  a2m FASTA; DCAlign.deltan_prior reads it to build Λ (§10.13)
#
# Output (appended to <out_tsv>, one row per query, flushed per row so a killed
# shard leaves a valid partial cache — the resume contract):
#   seq_id <TAB> aligned_frame <TAB> dcalign_energy <TAB> converged
#          <TAB> used_decimation <TAB> n_iter
# aligned_frame is the length-L frame as an amino-acid string (gap '-'),
# already in our alphabet (DCAlign decodes matches to the query letters). A
# per-sequence DCAlign error is recorded as an EMPTY frame + NaN energy and the
# driver continues (record-and-continue, no silent drop); an empty frame is a
# loud error at score time only if that id is needed.

using DCAlign
using OffsetArrays
using LinearAlgebra

const TSV_HEADER = "seq_id\taligned_frame\tdcalign_energy\tconverged\tused_decimation\tn_iter"


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


"""Build the insertion prior Λ (combine spec §10.13).

`"flat"`: mass on Δn=1 (adjacent match, no insertion) for every (i,j) pair — a
geometry-blind prior. The DCAlign `Alg` constructor reshapes this to (L,L,N+2),
adds the pcount floor plus small noise, and renormalises.

`"deltan"`: the empirical per-(i,j) prior `DCAlign.deltan_prior` builds from a
model-frame seed alignment (`seed.ins`, written by the Python bridge from each
model's own seed MSA). It returns `(Λ, Mseed, dist)`; we take Λ and hand it to
`palign` directly — exactly DCAlign's own `Run_alignment.jl` usage. The seed
carries no insert columns, so this learns the gap/deletion geometry, replacing
the flat prior's geometry-blind mass with real per-position statistics.
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
            error("lambda_spec=\"deltan\" needs $seedins (written by align_sequences)")
        return first(DCAlign.deltan_prior(seedins, L))
    end
    error("lambda_spec must be \"flat\" or \"deltan\", got \"$spec\"")
end


"""Align one raw query; returns the TSV row tuple."""
function align_one(header, rawseq, J, h, Λ, L::Int, maxiter::Int, seed::Int, pcount::Float64)
    seq = Seq(String(header), String(rawseq), :amino)
    niter, conv, res, _ = palign(seq, deepcopy(J), deepcopy(h), deepcopy(Λ), :amino;
                                 maxiter=maxiter, seed=seed, pcount=pcount, verbose=false)
    P = copy(res.pbf.P)
    out = DCAlign.decodeposterior(P, res.seq.strseq, thP=res.alg.thP)
    used_decimation = false
    # Nucleation fallback when the short-range constraints aren't satisfied.
    if !DCAlign.check_assignment(P, false, length(res.seq.strseq))
        _, P = DCAlign.decimate_post(res, false)
        out = DCAlign.decodeposterior(P, res.seq.strseq, thP=res.alg.thP)
        used_decimation = true
    end
    frame = out.seq
    length(frame) == L ||
        error("DCAlign frame length $(length(frame)) != L=$L for $header")
    energy = DCAlign.compute_potts_en(J, h, frame, L, :amino)
    converged = (conv == :converged)
    return (String(header), frame, energy, converged, used_decimation, Int(niter))
end


function main()
    length(ARGS) >= 2 || error("usage: run_dcalign.jl <in_dir> <out_tsv>")
    in_dir = ARGS[1]
    out_tsv = ARGS[2]

    meta = read_meta(joinpath(in_dir, "meta.json"))
    L = Int(meta["L"])
    q = Int(meta["q"])
    maxiter = Int(meta["maxiter"])
    seed = Int(meta["seed"])
    pcount = Float64(meta["pcount"])
    lambda_spec = String(meta["lambda_spec"])

    J, h = read_model(in_dir, q, L)
    Λ = build_lambda(lambda_spec, L, in_dir)
    queries = read_fasta(joinpath(in_dir, "queries.fasta"))
    println(stderr, "run_dcalign: L=$L q=$q maxiter=$maxiter seed=$seed pcount=$pcount " *
                    "lambda=$lambda_spec n_queries=$(length(queries)) threads=$(Threads.nthreads())")

    # Parallelise over the shard's sequences across JULIA_NUM_THREADS (set by the
    # sbatch wrapper from --cpus-per-task). align_one deepcopies J/h/Λ per call and
    # palign is seeded per sequence, so the per-row results are independent of the
    # thread count and of completion order — threading changes speed, not answers.
    # Rows are written under a lock and flushed per row, preserving the resume
    # contract (a killed shard leaves a valid partial cache, any order). BLAS is
    # pinned to one thread when we thread, to avoid oversubscribing the cores.
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
                align_one(header, rawseq, J, h, Λ, L, maxiter, seed, pcount)
            catch e
                @warn "DCAlign failed; recording empty frame" header exception = (e, catch_backtrace())
                (String(header), "", NaN, false, false, 0)
            end
            sid, frame, energy, converged, used_decimation, niter = row
            lock(iolock) do
                println(io, join((sid, frame, energy, converged, used_decimation, niter), '\t'))
                flush(io)
            end
        end
    end
    println(stderr, "run_dcalign: done -> $out_tsv")
end

main()
