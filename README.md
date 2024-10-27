# online min-max center problem.
## Instructions
1. Install the required packages (you might already have them in your base environment):
```bash
pip3 install -r requirements.txt
```

2. You will need a [gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/) for the MIP based offline problems.

3. Run the following commands to run the solvers on the synthetic and DHL datasets with the default arguments:
```bash
python3 run_experiments.py --setname unitsquare
```
```bash
python3 run_experiments.py --setname northeastdhl
```

4. You can run more complex experiments with different arguments, such as changing the fixed cost, changing the permutation (arrival order of the online demand points), etc. Run the following command to see the options:
```bash
python3 run_experiments.py --help
```

## Instance Data (OR Paper Submission):
Instances are saved in the `dat` directory, at `dat/<setname>/<setname>_n_T.csv`.

### Instance Format
The instances come in two different formats. First, the 'points' format, which is how the synthetic instances are encoded, lists the points on each line of a csv file. A 'point' is listed on a line with each coordinate separated by a space. For this format, we also encode a default fixed cost with a single float on the first line before the points, though this can be overriden when you run an experiment.

For the real dataset, we use a 'distance' format, since we have to pre-process the latitudes and longitudes with the help of Google's Road Distance API. The csv file for an instance will thus be a square matrix with the distances between each point.

The solver code will automatically detect which format it is reading!

### Synthetic Experiment

The testbed for section 5.1 is located at `dat/unitsquare`. This testbed is generated by the code in `testbed.py`. If you wish to use similar distributions to generate more synthetic instances, you can modify the file `testbed.py` (to choose different parameters for `T` and `n`), and then run it as follows:
```bash
python3 testbed.py --set_name newsetname
```
This will create a directory `dat/newsetname` with the instances generated with your parameters. Of course, you can adjust the `--set_name` argument to name this testbed however you like.

Additionally, the testbed for the MIP timestudy can be found at `dat/timestudy`. The instances generated in this set have 1000 demand points each - don't run the MIPs on the entire instances (all 1000 points), or you will wait an unreasonable amount of time for them to return solutions. See Appendix B for an overview of the limitations of the MILP models - we'd recommend trying out running only the first 50 points:
```bash
python3 run_experiments.py --set_name timestudy --T 50
```

### DHL Facilities Experiment
The testbed for section 5.2 is located at `dat/northeastdhl`.

## Notes on final.csv schema:
For permuations: you can pass none, start, end, full to the permutation argument. When we store the results in final.csv, we store the permuation in the perm_order column, and the indices are the indices of the original ordering. All other columns that use indices of points, such as the facilities column or facilities file, are stored in the permuted ordering, so they are a function of the permutation.

As an example, consider an instance with T = 2, so points 0, 1, 2. First we run it with --perm null, so we use the original ordering on-disk. The optimal solution turns out to build facilities 0 and 1. Some columns in final.csv:

perm | perm_order | facilities |
none | 0-1-2 | 0-1 |

Next we run it with --perm full, and the permutation sampled randomly turns out to be 2-0-1. The optimal solution is now to build facility 2 which arrives at time period 0 in this permutation. Some columns in final.csv:

perm | perm_order | facilities |
full | 2-0-1 | 0 |

Notice that the facilities column is still 0, because it is a function of the permutation. The facilities file follows the same pattern.
