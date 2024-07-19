**Dedicated platform for data mining!** 

_What is in here?_

This project contains a Streamlit platform where you can upload a tabular file (csv, xls, txt, data), and perform various analysis depending on your dataset. 

_What are the functionalities?_

You have 6 pages inside the Streamlit, each offering different functionalities: 

**Data import**

Import your file, specify the used delimiter and if there is a header or not. You can then have a look at a preview of your imported dataset to make sure everything went fine. 

_Please note that the next functionalities required you to have uploaded your file on the first page_

**Data exploration**

Discover the basic statistics of your dataset. Number of columns, rows, na values, as well as various basic statistical values (mean, std, ...)

**Data cleaning & normalization**

Clean your dataset using various NA handling methods (delete, replace with basic statistic method , replace with advanced algorithms). 
You can also normalize your data using various normalization methods (MinMax, Z score, MaxAbs). 
Please note that the cleaning/normalization process directly affects your uploaded dataset. So be careful when applying those modifications.

**Data visualization**

Create graphs to discover information & insights in your dataset. For example, you can study distribution of your variables using histograms. 

**Clustering or Prediction**

Now that you got a good grasp of your dataset, it is time to train some machine learning models! The platform handles 4 models: 
+ KMeans clustering (unsupervised)
+ DBSCAN clustering (unsupervised)
+ Linear regression (supervised/regression tasks)
+ Random forest classifier (supervised/classification tasks)

**Learning evaluation**

Once your model is trained, you might want to learn about its performances. Depending on the model you trained, the platform will display various evaluation criterias & graphs, like for example Davis-Bouldin score for KMeans clustering, or R2 score for regression problems.


_Great! How do I install all of this?_ 

Please follow the procedure to install and use our platform: 

+ Clone the git repository
+ Make sure to install all the libraries inside requirements.txt file
+ Go into your terminal (make sure that you are on the project directory)
+ Execute the following command: streamlit run landing_page.py
The Streamlit app should now run on your web browser! 
