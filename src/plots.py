
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

### steps vs utility for different k

#sudo python3 ex1-pokec-2.py 9 0

mapping_ = {}
for x in glob.glob("../out/exp1/*"):
    key = x.split("/")[-1]
    if len(glob.glob(x + "/*" )) == 3:
        mapping_[key] = glob.glob(x + "/*" )

# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)



if False:
    for out_fn in sorted(mapping_):
        for filename in mapping_[out_fn]:
            df = pd.read_csv(filename, sep="\t", header=None, index_col=None)
            df.columns = ["step", "f_S", "time", "readable"]
            
            
            df["time"] = [int(round(x,0)) for x in list(df["time"])]
            df.set_index("time", inplace=True)
            df.index = df.index - df.index[0]
            #df["f_S"] = np.log(df["f_S"]+1) 
            df.index = np.log(df.index+1)
            df.index.name = "Time (s)"
            #df = df.iloc[1:]
            #df = df[df.index % 5 == 0]
            
            #df.set_index("step", inplace=True)
            #df = df[df.index % 5 == 0]
            if "fairGreedy" in filename:
                label = "GREEDY"
                c = "r"
            if "LS" in filename:
                label = "MP-StreamLS"
                c = "y"
            if "fairMulti" in filename:
                label = "MP-FSM"
                c = "b"
            
            df["f_S"].plot(
                style="-", 
                label=label, 
                legend=True,
                drawstyle='steps-post',
                fontsize=10,
                c = c,
                alpha=0.5
                )
        #plt.title(title)
        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel("$f(S)$", fontsize=15)
        plt.tight_layout()
        plt.show()
        plt.savefig("../plot/exp1/" + out_fn + ".pdf", format="pdf")
        plt.clf()


if True:
    for graphname in ["pokec-gender", "pokec-age", "ig", "synth1"]:
        GROUP = glob.glob("../out/exp1/%s*"%graphname)

        for fairness_constraint in ["PR", "ER"]:
            one_lst = [x for x in GROUP if fairness_constraint in x]
            map_res = {}
            for fn in one_lst:
                for algos_res in glob.glob(fn + "/*"):
                    df = pd.read_csv(algos_res, sep="\t", header=None, index_col=None)
                    if len(df) <= 1:
                        continue
                    f_S = df.iloc[-1][1]
                    time_ = int(df.iloc[-1][2]) - int(df.iloc[1][2])
                    if "-" in graphname:
                        k_ = fn.split("/")[-1].split("-")[2][1:]
                    else:
                        k_ = fn.split("/")[-1].split("-")[1][1:]
                    k_ = int(k_)

                    algoname = algos_res.split("/")[-1][:-4]
                    if algoname not in map_res:
                        map_res[algoname] = pd.DataFrame(columns=["$f(S)$", "Time (s)"])
                    else:
                        map_res[algoname].loc[k_] = f_S, time_
                        map_res[algoname].sort_index(inplace=True)
                
            
            
            fig, ax1 = plt.subplots()



            label = "GREEDY"
            c = "black"
            if "greedy" in map_res:
                map_res["greedy"]["$f(S)$"].apply(lambda x: np.log10(x)).plot(
                        label=label, 
                        alpha=.5, 
                        c=c,
                        style="v--",
                        ax=ax1)


            label = "RANDOM"
            c = "green"
            if "random" in map_res:
                map_res["random"]["$f(S)$"].apply(lambda x: np.log10(x)).plot(
                        label=label, 
                        alpha=.5, 
                        c=c,
                        style="x--",
                        ax=ax1)



            label = "FAIR-GREEDY"
            c = "r"
            map_res["fairGreedy"]["$f(S)$"].apply(lambda x: np.log10(x)).plot(
                    label=label, 
                    alpha=.5, 
                    c=c,
                    style="o--",
                    ax=ax1)
            
            label = "MP-StreamLS"
            c = "y"
            map_res["MultiPassStreamLS"]["$f(S)$"].apply(lambda x: np.log10(x)).plot(
                    label=label, 
                    alpha=.5, 
                    c=c,
                    style="+--",
                    ax=ax1)

            label = "MP-FSM"
            c = "b"
            map_res["fairMultipass"]["$f(S)$"].apply(lambda x: np.log10(x)).plot(
                    label=label, 
                    alpha=.5, 
                    c=c,
                    style="*--",
                    ax=ax1)

            ax1.legend()
            ax1.set_ylabel("$f(S)$")
            ax1.set_xlabel("K")

            plt.savefig(
                "../plot/multipass/" + graphname + "-" + fairness_constraint + "-f_S" ".pdf",
                format="pdf"
                )


            plt.clf()

            fig, ax1 = plt.subplots()


            label = "GREEDY"
            c = "black"
            if "greedy" in map_res:
                map_res["greedy"]["Time (s)"].apply(lambda x: np.log10(x)).plot(
                    label=label, 
                    alpha=.5, 
                    c=c,
                    style="v--",
                    ax=ax1)

            """
            label = "RANDOM"
            c = "green"
            if "random" in map_res:
                map_res["random"]["Time (s)"].apply(lambda x: np.log10(x)).plot(
                    label=label, 
                    alpha=.5, 
                    c=c,
                    style="x--",
                    ax=ax1)

            """


            label = "FAIR-GREEDY"
            c = "r"
            map_res["fairGreedy"]["Time (s)"].apply(lambda x: np.log10(x)).plot(
                    label=label, 
                    alpha=.5, 
                    c=c,
                    style="o--",
                    ax=ax1)
            
            label = "MP-StreamLS"
            c = "y"
            map_res["MultiPassStreamLS"]["Time (s)"].apply(lambda x: np.log10(x)).plot(
                    label=label, 
                    alpha=.5, 
                    c=c,
                    style="+--",
                    ax=ax1)

            label = "MP-FSM"
            c = "b"
            map_res["fairMultipass"]["Time (s)"].apply(lambda x: np.log10(x)).plot(
                    label=label, 
                    alpha=.5, 
                    c=c,
                    style="*--",
                    ax=ax1)

            ax1.legend()
            ax1.set_ylabel("$\log_{10}(time)$")
            ax1.set_xlabel("K")

            plt.savefig(
                "../plot/multipass/" + graphname + "-" + fairness_constraint + "-time.pdf",
                format="pdf"
                )
            plt.clf()



for mode in ["ER", "PR"]:
    unique_df = pd.DataFrame()
    for graphname in ["pokecgender", "pokecage", "ig", "synth1"]:
        GROUP = glob.glob("../out/exp3/%s*"%graphname)
        main_df = pd.DataFrame(columns=["epsilon", "f(S)"])
        for fn in GROUP:
            if mode in fn:
                eps = float(fn.split("-")[-1][:-4].replace("eps",""))
                df = pd.read_csv(fn, sep="\t",  header=None)

                eps = round(eps,2)
                f_S = df.iloc[-1][1]
                main_df.loc[len(main_df)] = [eps, f_S]
        main_df.set_index("epsilon", inplace=True)
        main_df.sort_index(inplace=True)

        main_df.columns = [graphname]
        main_df = main_df/main_df.max()

        #main_df[graphname] = main_df[graphname].apply(lambda x: np.log10(x))

        #main_df.plot()
        #plt.show()

        unique_df = pd.concat([unique_df, main_df], axis=1)


    unique_df.rename(columns={
        "ig": "IG",
        "pokecage": "POKEC-A",
        "pokecgender": "POKEC-G",
        "synth1": "SYNTH"
        }, inplace=True)
    
    
unique_df.index.name = r"$\epsilon$"
unique_df.plot(linestyle="--", style= "o", fontsize=15)
plt.xlabel(r"$\varepsilon$", fontsize=20)
plt.ylabel("$f(S)$",fontsize=20)
plt.tight_layout()
plt.savefig("../plot/exp3/k300.pdf", format="pdf")


