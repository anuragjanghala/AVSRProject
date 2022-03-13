# AVSRProject

## Lip Reading from Speechless Videos using Audio-Visual Speech Recognition Technology

The purpose of this project was to build a simple way of fetching information from speechless videos that are harder to understand for a normal human being to analyze.

We live in a world full of technologies but there are some chances that those technologies might fail at some stage in their deliverable-times.

• when there is noise on the audio channel which makes it difficult to hear a person's 	audio.
• when an audio file of some video gets corrupted.
• when a person is physically challenged to talk like a normal person, which makes it difficult for that person to live life normally.

### Tools used:

PyTorch:
Pytorch is one of the most used Frameworks for research pupose as it is widely used by researchers in deep learning or machine learning fields.

NumPy:
A library for the python programming language, adding support for large, multi-dimensional arrays and matrices.

OpenCV:
A python library of programming functions mainly aimed at the real-time computer vision. It is used to caputuring video data and frames from the  input video using cv2.VideoCapture().


### Dataset Used:
LRW (Lip Reading in the Wild). (others can be used like LSR3-TED, GRID CORPUS)



### Implementation:
- 3D CNN
- RESNET-34
- BGRU
- SOFTMAX


### Evaluation:

There are between 800 and 1000 sequences for each word in the training set and 50 sequences in the validation and test sets.
Which makes 400,000 training files and 25,000 validation and 25,000 testing files



#### For 2 epochs and batch_size = 5 of training full dataset

##### While train Phase
----------
Epoch 1/2
Current Learning rate: [0.0003]
train Epoch:	 1	Loss: 4.1060	Acc:0.0806

val Epoch:	 1	Loss: 2.5287	Acc:0.3330

----------
Epoch 2/2
Current Learning rate: [0.0003]
train Epoch:	 2	Loss: 2.2312	Acc:0.3972

val Epoch:	 2	Loss: 1.4127	Acc:0.6014


##### While Test Phase

*** model has been successfully loaded! ***

val Epoch:	 1	Loss: 2.1316	Acc:0.6400

test Epoch:	 1	Loss: 2.2750    Acc:0.6600




#### After 30 epochs and batch_size = 36 of full dataset

##### While train Phase
----------
Epoch 36/36
Current Learning rate: [0.0001]
train Epoch:	 36	Loss: 0.01060	Acc:0.8264

val Epoch:	 36	Loss: 0.05287	Acc:0.8333


After 1 epoch and batch_size = 5 of full dataset

##### While Test Phase

*** model has been successfully loaded! ***

val Epoch:	 0	Loss: 0.5141	Acc:0.8400

test Epoch:	 0	Loss: 0.6177	Acc:0.8200



### Conclude:
In this work, I’ve presented an end-to-end visual based system which learns to extract features directly from the image pixels and performs classification using BGRUs on a class of 500 words. 

These results obtained from the model out performs most of the implemented technologies from past researches (like LipNet) on the LRW dataset. 

Next step would be to extend this system in order to be able to recognise sentences instead of isolated words. 



### Future Works:
- AI model production on Web Apps using nodejs, flask / fastAPI, pickle
- Dockerizing the project for the ease of access for other reseachers.