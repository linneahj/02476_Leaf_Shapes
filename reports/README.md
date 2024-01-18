---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [ ] Write unit tests related to model construction and or model training
* [ ] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [x] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

--- 48 ---

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

--- s194354, s153189, s231733 ---

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- question 3 fill here ---

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

--- question 4 fill here ---

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

--- question 5 fill here ---

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

--- question 6 fill here ---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

--- question 7 fill here ---

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- question 8 fill here ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

--- question 9 fill here ---

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did setup DVC for our raw data, although we did not utilize the version control functionality much, since no changes were made to the raw data. Initially we used gdrive as the remote storage, but for the later steps of the project, we also employed a GCP bucket. 

Since the ability to store big files in git repositories is limited (and also frowned upon), the raw data normally has to be downloaded from someplace else and placed in the correct folder, when setting up a new instance of the project. Having the raw data tracked using DVC, enables us to store the raw data in one place, and once the repository have been cloned, all the user have to do in order to get the data in the correct directory, is to do `dvc pull`, which is very convenient. However, since we did not change this data, dvc was somewhat under-utilized in the project, as the main appeal of dvc is specifically to do version control. In addition to that, the authentication required to use gdrive as remote storage, did introduce some added complexity to the later parts of the project, for example when running docker containers. Since this project was made on such a limited timeline, this added complexity was costly in the form of hours spent on this, rather than the deployment of models.  If we had had more time, it would also have been an obvious next step to store the models using dvc, since they are also quite large files, and dvc lets us store pointers to any large artifact files, not just data files.


### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

Our CI has been organized in 3 separate files; one makes use of ruff to check that we comply with PEP8, one is responsible for running the unittest related to the data, and one is for automatically building the docker image for training. Ideally, we would have implemented more unittests to also cover model construction and model training, but since we were pressed for time, we settled for testing the data only in this workflow. This workflow can be seen [here](https://github.com/linneahj/02476_Leaf_Shapes/blob/master/.github/workflows/tests.yml). Since this workflow handles the processing of data, it requires authentication to the google drive, where the raw data is stored. This is handled by github secrets, which only can be setup by the owner of the repository, which we learned the hard way, when one of us were off sick.  The workflow for triggering the building of a docker image is setup in a similar manner and can be seen [here](https://github.com/linneahj/02476_Leaf_Shapes/blob/master/.github/workflows/train_docker_build.yml). All workflows runs on ubuntu, and are triggered by pull requests to master/main. The master branch is protected, so although the workflow also trigger when pushing to it, the trigger should always be the pull request. In an ideal world, we would have liked to also employ other operating systems, in order to test that our workflow works independently of which operating system is used. We attempted to make use of caching, in order to not having to download the same packages over and over again,  but ended up removing it, as it didn’t function as well as we had hoped.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used the training script provided by the TIMM framework, which according to the [official docs](https://huggingface.co/docs/timm/main/en/training_script) has “a variety of training args”. This include number of epochs, input image size, number of classes to predict and whether to use wandb for logging. This means, that training can be easily configured using for example `python leaf_shapes/train_model.py ./data/processed/TIMM/ --model resnet18 --num-classes 99 --epochs 10  --img-size 64 -–log-wandb` to train a model for 10 epochs with wandb logging enabled. 
The most used version of this is hardcoded in the makefile, such that calling `make data` will result in both the making of training data and subsequently training a resnet18 model on it. 
For the data, we did set up hydra as well, though since it became obsolete for the training, it would probably have been nicer to just use a simple argparser for `make_data.py` as well instead, as that would have enabled us to streamline the make-file better.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

Since we used the training script provided by the TIMM framework, the training script automatically saved both it’s corresponding config-file and a log of training loss, evaluation loss, learning rate etc. for each epoch, together with checkpoints of the model during the last couple of epochs, as well as the current best performing model.  This is very useful, since it enables us to see, exactly what hyperparameters, our model was trained with. In order to reproduce an experiment, the parameters from the config-file can then be set to exactly the same. In an ideal world, the training script should be able to take the entire config-file as argument, to make it even easier to repeat an experiment, but we did not have time to look up, how to do that using TIMM. 
In order to save as much information as possible, we also used wandb for logging, and in addition to logging training loss, validation loss etc, it also saves a requirement file with all packages needed to rerun the experiment, as well as a yaml-file with the corresponding conda enviroment.
It would have been nice to also save the configuration of the data processor (which in practice just meant the size, the images were resized to, which could also be seen from the size of the model), but this information was also saved both by wandb and by TIMM, so the most elegant solution would probably be to not use hydra at all.


### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

A staple in graphs to be logged in machine learning, is the loss graph, as can be seen in the image below (update to a longer training run if time). Logging loss is important, as that shows us how well the model is learning. When also taking the validation loss into account, we can also use the graph to look for signs of overfitting. 



Another example of logged data specific to this project is the learning rate. Since we are using a TIMM resnet18 which we haven’t spend much time optimizing, we have kept the standard configuration for the learning rate. The learning rate is therefore not static, but uses both a decay rate for the learning rate and a warm-up learning rate. Having never used a non-static learning rate before, it is therefore also relevant to track the learning rate. 

In general, many different metrics can be relevant for the project, depending on the goal of the project. Our project for example logs both validation accuracy for the target matching the models top 1 prediction and for target being in the top 5 most likely species. Our data set includes subspecies such as Tilia Oliveri and Tilia Platyphyllos, so accuracy for the top 5 may be relevant, if the model for example have trouble telling subspecies apart, but still chooses correctly within the broader category. 
Apart from hyperparameters and metrics such as accuracy being worth tracking, wandb also offers support to track system variables. If for example our model is so big, it requires downsampling of the images to run on our GPU, it would be relevant to track RAM usage for the training of our model. This could also be relavant, if we needed to know, what it what require to train the model on another machine. 
The image below shows an example of a wandb report made during the project.
![id](https://github.com/linneahj/02476_Leaf_Shapes/blob/master/reports/figures/Screenshot_20240118_220327.png)

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

The creation of a docker image for model training is included in the CI workflow, and enables us to train models using our default hyper-parameters in a docker container instead of setting up an environment, which makes the training reproducible. The docker file for the training can be found [here](https://github.com/linneahj/02476_Leaf_Shapes/blob/master/dockerfiles/train_model.dockerfile)  and can be build to an image using `docker build -f dockerfiles/train_model.dockerfile . -t trainer_docker:latest` (or it can be pulled from dockerhub using ` docker pull linneahj/02476_leaf_project:<tag for latest build>` or from gcp using `docker pull gcr.io/leaf-shapes-02476/train`). Once the image has been created, it can be run as a container using `docker run --name experiment1 trainer_docker:latest`. 
When we were using google drive instead of GCP Buckets, authentication was needed to run the container (we just saved the authentication as an environment variable using the `-e` flag in the `docker run` command), but using data storage in google cloud makes the above command much simpler.
We also have a docker file for the API [here](https://github.com/linneahj/02476_Leaf_Shapes/blob/master/dockerfiles/model_api.dockerfile). Since the image contains a copy of an already trained model, this means the "prediction app" can be launched ready for use simply by running the container, and it is also the basis for cloud deployment of the model.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Debugging was performed in various ways. The old school print statement was often used, and on occasion also the build in debugger is VS Code. Often the starting point would be the error message and then working backwards from there, reading the documentation of TIMM or Docker or whatever framework was giving us trouble. We did unfortunately not have time to do any profiling of our code.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

Our GCP container registry is shown here: ![id](https://github.com/linneahj/02476_Leaf_Shapes/blob/master/reports/figures/gcp_container_registry.png)

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 22 fill here ---

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- question 24 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

There were many struggles in this project. We got off to a rocky start, as we were too ambitious and the data set and model, we initially choose, turned out to be too big to run locally on our pcs, and so we had to start over from scratch. The main source of frustration however was GCP, where many, many hours were spend in vain. Docker also introduced some troubles, especially with the authentication-issues. For this, the internet was a great help, although many hours still were spend with not much progress to show. We also had a bug resulting in models of the wrong size, which was very hard to pin down, as none of the usual debug tools were of much help there. 
On a more practical level, illness meant at least one of us was off sick almost every project day, with first one person falling sick, then another, making the workload even bigger. Over all, the motto of this project has been “Slow Progress!”

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 27 fill here ---
