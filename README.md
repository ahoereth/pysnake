# PySnake - a TensorFlow project

This is a project to accompany the course "Implementing Artificial Neural Networks with TensorFlow".

We implement a simple snake game and different playing strategies:

- **Human controlled**: `python human_snake.py [--store subjectID]` runs the snake to be controlled with the arrow keys. Supply a subject ID to store the score in `participant[subjectID].csv`. If no ID but `--store` is supplied, it is stored in `participant_unspecified.csv`, to not lose it.
- **Systematic**: `python systematic_snake.py [--store]` runs the systematic snake. Supply `--store` to store the result into `systematic.csv`.
- **Q-Learning**: `python q_snake.py` -- details follow
- **Evolve**:Use `python evolve_snake.py train` if you want to start a new training session with the parameters defined in the class or use `python evolve_snake.py play <file>.np` to replay a given snake network with the correct number of weights.

For the accompanying project report, check [overleaf](https://www.overleaf.com/8169167cgrvhvrwfbqs).


## Videos
You can add `--video [name.mp4]` to save a movie. If you do not supply a name, it's stored as `pysnake.mp4`. This is not perfect yet, but it's there. Note that if you use this in conjunction with `--store subject` you should fully specify all but the last argument:

    python human_snake.py --video human.mp4 --store
    python human_snake.py --store 123 --video
    python human_snake.py --video human.mp4 --store 123

Similarly you should use `--store` before `--video` or supply a file name for the systematic snake:

    python systematic_snake.py --store --video
    python systematic_snake.py --video systematic.mp4 --store


## Shortcuts
To aggregate n runs of data of the systematic snake, run:

    make sys n=100
    make sys n=1000

To run a participant, just run:

    make id=1


## awstf.py
The `awstf.py` script looks for the cheapest AWS region to lunch a `p2.xlarge` spot instance and provides you with a `docker-machine` command to launch such an instance prepared with an AMI optimized for Tensorflow GPU computing using `nvidia-docker` and Google's official Tensorflow container image.

After launching a container scp your files to the container (or use git ssh'ed into the container) and start training:

```
localhost$ eval $(docker-machine env machinename)
localhost$ docker-machine scp ./yourscript.py machinename:/home/ubuntu/
localhost$ docker-machine ssh machinename
awsremote$ sudo nvidia-docker run -it -v /home/ubuntu:/workdir tensorflow/tensorflow:latest-gpu-py3 python3 /workdir/yourscript.py
```
Or if you want to run the `evolve_snake.py` use the following command after copying over the relevant source files:
```
sudo docker run -it -v /home/ubuntu:/workdir tensorflow/tensorflow:latest-py3 python3 /workdir/evolve_snake.py
```
