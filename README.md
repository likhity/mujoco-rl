# Reinforcement Learning for Robotic Pushing and Bowling in Mujoco

![image](/bowling.gif)

## See For Yourself

After you clone this repo, we recommend using a virtual environment. Create one like so:

```
python -m venv venv
```

Then activate the virtual environment:

```
source venv/Scripts/activate
```

Make sure the virtual environment is always activated.

Now, install all necessary packages:

```
pip install -r requirements.txt
```

This project essentially consists of two main parts. 

1. Box pushing with a two-link arm
2. Bowling with a 7DOF pusher

### Part 1

We initially did Part 1 with the initial location of the box fixed in one place. The final model for this task in the **`master`** branch. So, please make sure you are in the master branch to view the result of this task. To do this, run:

```
git checkout master
```

To view its behavior, simply run:

```
python watch.py
```

This will open the Mujoco simulation.

We then wanted to randomize the initial location of the box to promote generalization.

To see the result of this, switch to the **`random`** branch.

```
git checkout random
```

Now run `python watch.py` to see the result of this task where we randomized the initial location of box.

### Part 2

Switch to the **`bowling`** branch.

```
git checkout bowling
```

Now run `python watch.py` to see the result of the bowling task.