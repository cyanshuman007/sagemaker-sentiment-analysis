# IMDB sentiment analysis - AWS SageMaker

A random tree model to predict the sentiment of a movie review using the XGBoost package and AWS SageMaker.
The dataset used is the IMDb dataset. It consists of movie reviews from the website imdb.com, each labeled as either 'positive', if the reviewer enjoyed the film, or 'negative' otherwise.

Download Dataset at - http://ai.stanford.edu/~amaas/data/sentiment/

This model had an accuracy of 87% when applied to the dataset available on 01/12/2020.

An AWS Account with permission to use Amazon SageMaker is required.
1)	The first thing to do is set up a notebook instance which will be the primary way in which we interact with the SageMaker ecosystem.
2)	Log in to the AWS console, open the SageMaker dashboard and selecting Notebook Instances, then click on ‘Create notebook instance’.
3)	Choose any name you would like for your notebook. Use a ml.t2.medium for the notebooks as it is covered under the free tier.
4)	Next, under IAM role select ‘Create a new role’ and select None under ‘S3 buckets you specify’.
5)	Now scroll down and click on Create notebook instance.
6)	Once your notebook instance has started and is accessible, click on open to get to the Jupyter notebook main page.

In order to clone the deployment repository into your notebook instance, click on the new drop down menu and select terminal. By default, the working directory of the terminal instance is the home directory, however, the Jupyter notebook hub's root directory is under SageMaker. Enter the appropriate directory and clone the repository as follows:
cd SageMaker
git clone https://github.com/cyanshuman007/sagemaker-sentiment-analysis.git
exit

After you have finished, close the terminal window.
Your notebook instance is now set up and ready to be used!
