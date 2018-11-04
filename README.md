# PolarBears

With elections just around the corner, we thought it would be interesting to examine the political stances of various US politicians. Using the VADER sentiment analysis tool, we assigned scores to various politicians based on tweets that contained certain contentious issues. Then, using Principal Component Analysis and k-means clustering, we transformed and clustered the data. Finally, we built an interactive website using Dash and Plot.ly and deployed it using Heroku.

## Visualize
[Take a look at our website!](https://powerful-citadel-26548.herokuapp.com/)


## Built With

* [vaderSentiment](https://github.com/cjhutto/vaderSentiment) - Sentiment analysis tool
* [Pandas](https://pandas.pydata.org) - Data manipulation
* [Scikit Learn](scikit-learn.org) - Machine learning library
* [Dash](https://dash.plot.ly/installation) - Web framework
* [Plot.ly](https://plot.ly) - Data visualization


## Team

* **Anup Hiremath** - [Github](https://github.com/anup-h)
* **Jeffrey Liu** - [Github](https://github.com/franklinfrank)

## Acknowledgements

* [Tweet Preprocessing Tool](https://github.com/s/preprocessor) used to clean tweets before analysis
* [Paper on analyzing political sentiment on Twitter](https://www.aaai.org/ocs/index.php/SSS/SSS13/paper/download/5702/5909)
* [Study on linguistic indicators of bias in politics](http://www.cs.cmu.edu/~nasmith/papers/yano+resnik+smith.wamt10.pdf)
