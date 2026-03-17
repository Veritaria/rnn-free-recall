# sacctmgr modify user ml8736 set defaultaccount=torch_pr_279_tandon_priority

gamma=('0' '02' '04' '06' '08' '10')
noise=('0' '02' '04' '06' '08' '1')

ctx_noise=('0' '02' '04' '06' '08')
# ctx_noise=('0' '04' '08' '12')

# for n in "${noise[@]}"
# do
#     for g in "${gamma[@]}"
#     do
#         # python run_cluster.py --exp VaryAllSeq8NoNoise --cpus_per_task 1 --setup setup_gamma${g}_noise${n}.json --time 1 --exp_file item_invariant
#         python run_cluster.py --exp VaryAllSeq8LargeNoise --cpus_per_task 1 --setup setup_gamma${g}_noise${n}.json --time 1
#         python run_cluster.py --exp VaryAllSeq8 --cpus_per_task 1 --setup setup_gamma${g}_noise${n}.json --time 1
#         python run_cluster.py --exp VaryAllSeq8NoNoise --cpus_per_task 1 --setup setup_gamma${g}_noise${n}.json --time 1
#     done
# done

for n in "${ctx_noise[@]}"
do
    # python run_cluster.py --exp ExtraObs.EnvCxt --cpus_per_task 1 --setup setup_gamma06_randnoise${n}.json --time 10 --mem 16 -train
    python run_cluster.py --exp ExtraObs.EnvCxt --cpus_per_task 1 --setup setup_gamma09_flush09_gaussiannoise${n}.json --time 1 --mem 16
    python run_cluster.py --exp ExtraObs.EnvCxt --cpus_per_task 1 --setup setup_gaussiannoise${n}.json --time 1 --mem 16
    # python run_cluster.py --exp ExtraObs.EnvCxt --cpus_per_task 1 --setup setup_gamma1_flush09_dim41_gaussiannoise${n}.json --time 10 --mem 16 -train
    # python run_cluster.py --exp ExtraObs.EnvCxt --cpus_per_task 1 --setup setup_dim41_randnoise${n}.json --time 10 --mem 16 -train
done

# python run_cluster.py --exp ExtraObs.RecallCtx --cpus_per_task 1 --setup setup_gaussiannoise1_same.json --time 1 --mem 16
# python run_cluster.py --exp ExtraObs.RecallCtx --cpus_per_task 1 --setup setup_gaussiannoise1_diff.json --time 1 --mem 16
# python run_cluster.py --exp ExtraObs.RecallCtx --cpus_per_task 1 --setup setup_gaussiannoise1_baseline.json --time 1 --mem 16
# python run_cluster.py --exp ExtraObs.RecallCtx --cpus_per_task 1 --setup setup_randnoise1_baseline.json --time 1 --mem 16

# python run_cluster.py --exp Semantic --cpus_per_task 1 --setup setup_extra_hierarchy_amp05.json --time 1 --mem 16 --exp_file semantic
# python run_cluster.py --exp Semantic --cpus_per_task 1 --setup setup_extra_hierarchy_amp05_gamma0.json --time 1 --mem 16 --exp_file semantic
# python run_cluster.py --exp Semantic --cpus_per_task 1 --setup setup_extra_hierarchy_amp02.json --time 1 --mem 16 --exp_file semantic
# python run_cluster.py --exp Semantic --cpus_per_task 1 --setup setup_extra_hierarchy_amp02_gamma0.json --time 1 --mem 16 --exp_file semantic
# python run_cluster.py --exp Semantic --cpus_per_task 1 --setup setup_extra_hierarchy_amp05_gamma09_flush09.json --time 1 --mem 16 --exp_file semantic
# python run_cluster.py --exp Semantic --cpus_per_task 1 --setup setup_extra_hierarchy_amp02_gamma09_flush09.json --time 1 --mem 16 --exp_file semantic

# python run_cluster.py --exp ExtraObs.EnvCxt --cpus_per_task 1 --setup setup_gaussiannoise05_sigma05.json --time 1 --mem 16

# python run_cluster.py --exp ExtraObs.EnvCxt --cpus_per_task 1 --setup setup_randnoise05_dim41.json --time 1 --mem 16


# python run_cluster.py --exp ExtraObs.EnvCxt --cpus_per_task 4 --setup setup_randnoise1.json --time 11 -train

# squeue --me --format="%.18i %.9P %.80j %.8u %.2t %.10M %.6D %R"