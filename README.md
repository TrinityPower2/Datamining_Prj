<h1>Dedicated platform for data mining!</h1>

<h2>Git hub link:</h2>
https://github.com/TrinityPower2/Datamining_Prj.git

<h2>What is in here?</h2> 
<p>This project contains a Streamlit platform where you can upload a tabular file (csv, xls, txt, data), and perform various analysis depending on your dataset.</p>

<h2>What are the functionalities?</h2>
<p>You have 6 pages inside the Streamlit, each offering different functionalities: </p>

<h3>Data import</h3>
<p>Import your file, specify the used delimiter and if there is a header or not. You can then have a look at a preview of your imported dataset to make sure everything went fine. </p>

<h4> Please note that all the next functionalities require you to have uploaded your file on the first page! </h4>

<h3>Data exploration</h3>
<p>Discover the basic statistics of your dataset. Number of columns, rows, na values, as well as various basic statistical values (mean, std, ...).</p>

<h3>Data cleaning & normalization</h3>
<p>Clean your dataset using various NA handling methods (delete, replace with basic statistic method , replace with advanced algorithms). 
You can also normalize your data using various normalization methods (MinMax, Z score, MaxAbs). 
Please note that the cleaning/normalization process directly affects your uploaded dataset. So be careful when applying those modifications.</p>

<h3>Data visualization</h3>
<p>Create graphs to discover information & insights in your dataset. For example, you can study distribution of your variables using histograms.</p>

<h3>Clustering or Prediction</h3>
<p>Now that you got a good grasp of your dataset, it is time to train some machine learning models! The platform handles 4 models:</p>
<ul>
<li>KMeans clustering (unsupervised)</li> 
<li>DBSCAN clustering (unsupervised)</li>
<li>Linear regression (supervised/regression tasks)</li> 
<li>Random forest classifier (supervised/classification tasks)</li> 
</ul>
<h3>Learning evaluation</h3>
<p>Once your model is trained, you might want to learn about its performances. Depending on the model you trained, the platform will display various evaluation criterias & graphs, like for example Davis-Bouldin score for KMeans clustering, or R2 score for regression problems.</p>


<h2>Great! How do I install all of this?</h2>
<p>Please follow the procedure to install and use the platform:</p>
<ol>
<li>Clone the git repository</li> 
<li>Make sure to install all the libraries inside requirements.txt file (in a terminal: pip install -r requirements.txt)</li> 
<li>Go into your terminal (make sure that you are on the project directory)</li>
<li>Execute the following command: streamlit run landing_page.py</li> 
</ol>
<p>The Streamlit app should now run on your web browser!</p>
