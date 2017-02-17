# PySnake - a TensorFlow project

This is a project to accompany the course "Implementing Artificial Neural Networks with TensorFlow".

We implement a simple snake game and different playing strategies:

- Human controlled: `python human_snake.py [subjectID]` runs the snake to be controlled with the arrow keys. Supply a subject ID to store the score in `participant[subjectID].csv`.
- Systematic: `python systematic_snake.py [store]` runs the systematic snake. Supply some value as store to store the result into `systematic.csv`.
- Q-Learning: `python q_snake.py` -- details follow
- Evolve: `python evolve_snake.py` -- details follow

For the accompanying project report, check [overleaf](https://www.overleaf.com/8169167cgrvhvrwfbqs).



## awstf.py
The `awstf.py` script looks for the cheapest AWS region to lunch a `p2.xlarge` spot instance and provides you with a `docker-machine` command to launch such an instance prepared with an AMI optimized for Tensorflow GPU computing using `nvidia-docker` and Google's official Tensorflow container image.

After launching a container scp your files to the container (or use git ssh'ed into the container) and start training:

```
localhost$ eval $(docker-machine env machinename)
localhost$ docker-machine scp ./yourscript.py machinename:/home/ubuntu/
localhost$ docker-machine ssh machinename
awsremote$ sudo nvidia-docker run -it -v /home/ubuntu:/workdir tensorflow/tensorflow:latest-gpu-py3 python3 /workdir/yourscript.py
```
