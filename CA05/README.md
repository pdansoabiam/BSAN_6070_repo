CA05 – kNN Movie Recommender Engine

A kNearest Neighbors recommender system built with Python and scikitlearn that finds the 5 most similar movies to a given query movie based on genre and IMDB rating data.

Problem Statement
Given a movies dataset, find the **5 most similar movies** to a query movie ("The Post") using the kNN algorithm. This simulates the "More Like This" feature found on streaming platforms like Netflix or Hulu.


Dataset

 Property 	 Detail 										

 File 		 movies_recommendation_data.csv 							
 Source 	 UCI IMDB dataset (subset) 								
 Records 	 30 movies 										
 URL 		 https://github.com/ArinB/MSBACAData/raw/main/CA05/movies_recommendation_data.csv 	

Features Used

 Column 	 Type 		 Description 			

 IMDB Rating 	 Float 	 Movie rating (e.g. 7.2) 	
 Biography 	 Binary 	 1 = Yes, 0 = No 		
 Drama 	 Binary 	 1 = Yes, 0 = No 		
 Thriller 	 Binary 	 1 = Yes, 0 = No 		
 Comedy 	 Binary 	 1 = Yes, 0 = No 		
 Crime 	 Binary 	 1 = Yes, 0 = No 		
 Mystery 	 Binary 	 1 = Yes, 0 = No 		


Note: The Labels column is all zeros and is ignored. The History column mentioned in the instruction brief does not exist in the dataset and must be omitted from the query vector.


Requirements

bash
pip install pandas numpy scikitlearn


Project Structure


CA05/
README.md
pda_ah_CA05_kNN_based_Movie_Recommender_Engine.ipynb.ipynb       Main Jupyter notebook
movies_recommendation_data.csv					Autoloaded from GitHub URL

Key Design Decisions
Model: NearestNeighbors (not KNeighborsClassifier)
This is a recommender, not a classifier. We are finding similar items, not predicting a label. NearestNeighbors with .kneighbors() returns distances and indices — exactly what we need.

python
from sklearn.neighbors import NearestNeighbors

neigh = NearestNeighbors(n_neighbors=6)
neigh.fit(X)
distances, indices = neigh.kneighbors(query_vector)

Why n_neighbors=6 and not 5?
The query movie may exist in the dataset. Setting n_neighbors=6 and skipping indices[0][0] (the selfmatch at distance 0) ensures we always return exactly 5 meaningful recommendations.

Query Vector for "The Post"
python
IMDB=7.2, Biography=1, Drama=1, Thriller=0, Comedy=0, Crime=0, Mystery=0
the_post = pd.DataFrame([[7.2, 1, 1, 0, 0, 0, 0]], columns=feature_cols)


Notebook Workflow

 Cell 	 Purpose 						

 1 	 Import libraries 					
 2 	 Load dataset from GitHub URL 				
 3 	 Explore data (shape, columns, head) 			
 4 	 Prepare feature matrix X 				
 5 	 Fit NearestNeighbors model 				
 6 	 Define query vector for "The Post" 			
 7 	 Run .kneighbors() — get distances and indices 	
 8 	 Display top 5 recommended movie names 		
 9 	 Clean results DataFrame output 			


Usage
Open pda_ah_CA05_kNN_based_Movie_Recommender_Engine.ipynb in Jupyter and run all cells sequentially. The final output displays the 5 movies most similar to "The Post" along with their similarity distances.

To query a different movie, update Cell 6 with that movie's genre values and IMDB rating, then rerun Cells 7–9.



Output
Top 5 Movies Similar to 'The Post':

	Rank	Movie Name		Distance
0	1	The Wind Rises		0.6
1	2	12 Years a Slave	0.9
2	3	Hacksaw Ridge		1.0
3	4	A Beautiful Mind	1.0
4	5	The Karate Kid		1.0


Limitations
 Similarity is based solely on genres and IMDB rating. Factors like actors, directors, themes, and plot are not captured in this dataset.
 The dataset contains only 30 movies, so recommendations are limited to that pool.
 All features are treated with equal weight. No feature scaling is applied since binary features already exist on a 0–1 range alongside the IMDB rating.

References
 [scikitlearn NearestNeighbors](https://scikitlearn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)
 [UCI IMDB Dataset](https://archive.ics.uci.edu/ml/datasets/Movie)
 Course Assignment: CA05 – kNN Recommender Engine
