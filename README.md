# Hybrid Firefly–Particle Swarm Optimization (HF-PSO)
### Cloudlet Placement Optimization in Edge Computing

This project implements a **Hybrid Firefly + Particle Swarm Optimization algorithm**  
to solve the **Cloudlet Placement Problem** in heterogeneous Edge Computing environments.

The algorithm places cloudlets at candidate locations and assigns devices to them  
while minimizing:

- ✔ Total latency  
- ✔ Total placement cost  
- ✔ Constraint violations (coverage + capacity)  

The solver uses a hybrid approach combining:
- Firefly Algorithm (FA) movement toward brighter solutions  
- Discrete Particle Swarm Optimization (PSO) updates  
- Constraint repair operators  
- A simple non-dominated archive for cost–latency Pareto trade-offs  

---

## 📂 **Project Structure**

```
cloudlet_ffpso/
├── README.md
├── requirements.txt
├── run.py
├── data/
│   └── generate_synthetic.py
├── src/
│   ├── problem.py
│   ├── hybrid_ff_pso.py
│   ├── utils.py
│   └── experiments.py
└── examples/
    └── example_run.sh
```

---

# 🚀 **How to Run (Ubuntu / Debian / Linux)**

This version includes full instructions for systems where Python is **externally-managed**,  
which is why you may see:
> `ensurepip is not available`  
> `externally-managed-environment`  
> `python command not found`

## ✅ **1) Install venv (required once)**

Some systems do NOT include the venv module. Install it:

```bash
sudo apt install python3.12-venv
```

> If you use Python 3.10 or 3.11, replace `3.12` with your actual version.

---

## ✅ **2) Create a virtual environment**

From project root:

```bash
python3 -m venv venv
```

---

## ✅ **3) Activate the environment**

```bash
source venv/bin/activate
```

You should now see:

```
(venv) yourname@pc:~/project$
```

---

## ✅ **4) Install dependencies**

```
pip install -r requirements.txt
```

If your system still complains about "externally managed environment", run:

```
pip install --break-system-packages -r requirements.txt
```

---

## ✅ **5) Run the solver**

Run:

```
python3 run.py
```

or:

```
python run.py
```
(if python alias is available)

---

# 🧪 **Example Output**

You will see progress bars:

```
Iter 0: best fitness ...
Iter 10: best fitness ...
...
Finished. Archive size: X
```

At the end, the script prints:

- ⭐ Best solution metrics  
- ⭐ Placement of cloudlets  
- ⭐ Device assignments  
- ⭐ Pareto archive summary  

---

# 🧰 **Troubleshooting**

### ❌ `python: command not found`  
Use:

```
python3 run.py
```

### ❌ `ensurepip is not available`  
Install:

```
sudo apt install python3.12-venv
```

### ❌ `externally-managed-environment`  
Inside a venv this should not happen.  
But if needed:

```
pip install --break-system-packages -r requirements.txt
```

---

