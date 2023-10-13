rule Bu2JpsiK:
    input: Tuple = expand("/ceph/users/molocco/classical-taggers/Data/daVinci_scripts/Bu2JpsiK/RootRaw/Tuple_SM_{n}.root", n = range(7238)), Histos = expand("/ceph/users/molocco/classical-taggers/Data/daVinci_scripts/Bu2JpsiK/RootRaw/Histos_SM_{n}.root", n = range(7238))

rule run_Bu2JpsiK_DaVinci:
    input:  opt = "/ceph/users/molocco/classical-taggers/Data/daVinci_scripts/Bu2JpsiK/options_SM/options_{n}.yaml", dst = "/ceph/users/molocco/classical-taggers/Data/daVinci_scripts/Bu2JpsiK/output/hlt2_SM_{n}.dst",json = "/ceph/users/molocco/classical-taggers/Data/daVinci_scripts/Bu2JpsiK/output/hlt2_tck_SM_{n}.json"
    output: "/ceph/users/molocco/classical-taggers/Data/daVinci_scripts/Bu2JpsiK/RootRaw/Tuple_SM_{n}.root",  "/ceph/users/molocco/classical-taggers/Data/daVinci_scripts/Bu2JpsiK/RootRaw/Histos_SM_{n}.root"
    log: "/ceph/users/molocco/classical-taggers/Data/daVinci_scripts/Bu2JpsiK/logs/log_SM_DaVinci_{n}.txt"
    run:
        command = f"/interactive_storage/molocco/stack_v2/DaVinci/build.x86_64_v2-centos7-gcc12+detdesc-opt/run lbexec /ceph/users/molocco/classical-taggers/Data/daVinci_scripts/Bu2JpsiK/Algs.py:main {input.opt}|tee {log}"
        shell(command)


rule run_Bu2JpsiK_Moore:
    input: "/ceph/users/molocco/classical-taggers/Data/moore_scripts/SM_lines/Bu2JpsiK/line_SnakeMake_{n}.py"
    output: "/ceph/users/molocco/classical-taggers/Data/daVinci_scripts/Bu2JpsiK/output/hlt2_SM_{n}.dst","/ceph/users/molocco/classical-taggers/Data/daVinci_scripts/Bu2JpsiK/output/hlt2_tck_SM_{n}.json"
    log: "/ceph/users/molocco/classical-taggers/Data/moore_scripts/SM_lines/Bu2JpsiK/logs/SM_Moore_{n}.txt"
    run:
        command = f"/interactive_storage/molocco/stack_v2/Moore/build.x86_64_v2-centos7-gcc12+detdesc-opt/run gaudirun.py {input} | tee {log}"
        shell(command)
