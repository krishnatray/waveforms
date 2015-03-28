### Andrew Maner 
### Thinkful Data Science with Python Unit 5 (Capstone)

#### Introduction

Dataset: "Waveform Database Generator (Version 1)"

Downloaded from: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Waveform+Database+Generator+%28Version+1%29)

Data source (original):  Breiman, L., Friedman, J.H., Olshen, R.A., & Stone, C.J. (1984). *Classification and Regression Trees*. Wadsworth International Group: Belmont, CA (pp. 43-49)

Information:

* 3 classes of waves (0, 1, 2)
* 21 attributes, with noise
* 5000 instances
* Each generated from combination of 2 and 3 "base" waves
* Each with mean 0 variance 1 noise added

Goal: To find most efficient (in terms of features used and processor time) naive method to classify these waveforms, with a reasonable measure of accuracy.

#### Data Exploration

Step 1: Read the data into a 5000 x 22 pandas data frame.  Columns 1-21 contain features (and are labeled "fn", where n is the feature number), and column 22 contains the class ids (and is labeled "class").

Step 2: Set a seed and plot five (5) random combinations of waveforms, to look for commonalities and differences.  (More plots were visualized, but 5 seemed to summarize some important characteristics.)  Example:

```python
 x = range(1,22)
random.seed(0)
xind1 = random.randint(0,4999)
xind2 = random.randint(0,4999)
print "xind1: " + str(xind1) + ", class: " + str(df.loc[xind1, 'class'])
print "xind2: " + str(xind2) + ", class: " + str(df.loc[xind2, 'class'])
plt.figure()
plt.scatter(x, df.loc[xind1, colnames[0:21]], c='r')
plt.scatter(x, df.loc[xind2, colnames[0:21]], c='b')
plt.xlabel('X')
plt.ylabel('Y')
fig_title = "wave " + str(xind1) + " & wave " + str(xind2)
plt.title(fig_title)
plt.savefig('two_random_waveforms_1.png')
plt.clf()
```

First pairing - instances 4222 and 3789, both of class 2.  Note the clear similarites, despite the noise.

![alt text](https://github.com/amaner/waveforms/blob/master/two_random_waveforms_1.png)

Second pairing - instances 2102 and 1294, both of class 0.  Note again the clear similarities.

![alt text](https://github.com/amaner/waveforms/blob/master/two_random_waveforms_2.png)

Third pairing - instances 2556 and 2024.  These are from classes 0 and 1, respectively, and have clear differences *and* similarities.  

![alt text](https://github.com/amaner/waveforms/blob/master/two_random_waveforms_3.png)

Fourth pairing - instances 3918 and 1516.  These are from 0 and 2, respectively, and are very much different (would almost cancel).

![alt text](https://github.com/amaner/waveforms/blob/master/two_random_waveforms_4.png)

Fifth pairing - instances 2382 and 2916.  These are from classes 1 and 2, respectively, and have clear differences *and* similarities (as in the third pairing).

![alt text](https://github.com/amaner/waveforms/blob/master/two_random_waveforms_5.png)