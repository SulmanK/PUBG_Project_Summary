import base64
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


########################### Import Logo and Plots
image1_filename = 'assets/PUBG_4K_Logo.jpg' 
encoded_image1 = base64.b64encode(
    open(image1_filename, 'rb').read())

image2_filename = 'assets/Kmeans_Inertia.png' 
encoded_image2 = base64.b64encode(
    open(image2_filename, 'rb').read())


image3_filename = 'assets/Kmeans_Silhouette.png' 
encoded_image3 = base64.b64encode(
    open(image3_filename, 'rb').read())


image4_filename = 'assets/PCA_Optimum.png' 
encoded_image4 = base64.b64encode(
    open(image4_filename, 'rb').read())

image5_filename = 'assets/PCA_Variance.png' 
encoded_image5 = base64.b64encode(
    open(image5_filename, 'rb').read())



############# Import Figures
values  =  [['K-means Clustering (Clusters  =  2)',
           'K-means Clustering (Clusters  =  4)',
           'Density-based spatial clustering of applications with noise (DBSCAN)',
           'Local Outlier Factor (LOF)',
           'Elliptic Envelope (EE)',
           'Isolation Forest (IF)',
          ], #1st col
 
          ['53902.7243',
           '36075.5195',
           '-',
           '-',
           '-',
           '-'
          ], #2nd col
          
          ['0.4635',
           '0.2609',
           '0.7487',
           '0.7332',
           '0.7427',
           '0.7440'   
          ], # 3rd col
          
          ['Humans: 17225 | Hackers: 3546',
           'Beginners: 9059 | Experienced: 6561 | Professionals: 4332 | Hackers: 819',
           'Humans: 20689 | Hackers: 82',
           'Humans: 20650 | Hackers: 121',
           'Humans: 20650 | Hackers: 121',
           'Humans: 20650 | Hackers: 121'
           ], # 4th col
          
          
          ['Humans: 5898 Hackers: 1223',
           'Beginners: 3139 | Experienced: 2221 | Professionals: 1448 | Hackers: 313',
           'Humans: 7040 | Hackers: 83',
           'Humans: 7072 | Hackers: 49',
           'Humans: 7073 | Hackers: 48',
           'Humans: 7071 | Hackers: 50'
          ], # 5th col
          
          ['Humans: 1454 Hackers: 317',
           'Beginners: 766 | Experienced: 564 | Professionals: 380 | Hackers: 71',
           'Humans: 1773 | Hackers: 8',
           'Humans: 1768 | Hackers: 13',
           'Humans: 1770 | Hackers: 11',
           'Humans: 1769 | Hackers: 12'
          ], # 6th col
          ['Initialized the K-means clustering algorithm with two clusters based on the assumption, the population was divided into Humans and Hackers. The number of clusters was set to two because after comparing the number of clusters to the silhouette score, two clusters provided the greatest value.',
           'Initialized the K-means clustering algorithm with four clusters based on the assumption, the population was divided into Beginners, Experienced, Professionals, and Hackers. The number of clusters was set to four because after comparing the number of clusters to the inertia score, four clusters provided the greatest value.',
           'After analyzing the results on K-means, there was an abundant amount of "hackers" detected by the algorithm. Given a statistic that hackers represent 0.58% of the population, the problem was reframed as anomaly detection or an outlier detection problem. DBSCAN was the first of its kind to be implemented and produced promising results.',           
           'Local Outlier Factor was selected because of its recent success in Big Data. The algorithm compares the local densities around neighbors rather than the global densities. Then, weighing the relative density of an object against its neighbors as an indicator of the degree of the object being outliers. Similar to DBSCAN, LOF produced promising results, but suffered from a considerable amount of misclassification',
           'Elliptic Envelope was chosen because of its simple formulation in fitting multivariate gaussian densities to our data. And if our data was determined to be Gaussian, it would be a viable choice of action. After visual inspection, this algorithm proved to be the second-best performing choice. But I am skeptical in accepting the results, as the algorithm is not robust to outliers, so I would have expected different results.',
           'Finally, our quest in finding alternative anomaly detection algorithms ends with Isolation Forest. The algorithm isolates each point in the data and splits them into outliers or inliers. The split depends on how long it takes to separate the points. If the point is an outlier, it will be easy to split, but if the point is an inlier, it will be difficult to isolate. Isolation Forest performed the best out of all the algorithms. However, as expected there would be misclassification present in examining the clusters. Possible solutions to minimize the misclassification are parameter tuning, alternative anomaly detection algorithms, dimensional reduction techniques, or alternative KPIs to use.']
         ]


fig  =  go.Figure(data = [go.Table(
    columnorder  =  [1,2, 3, 4, 5, 6, 7],
    columnwidth  =  [200, 150, 150, 350, 350, 350,  400],
    header  =  dict(
        values  =  [['Method'],
                    ['Scoring Metrics', ['Inertia']],
                    ['', ['Silhouette ']],
                    ['Results', ['Training Set']],
                    ['', ['Dev Set']],
                    ['', ['Testing Set']],
                    ['Description']],
        line_color = 'darkslategray',
        fill_color = '#90ee90',
        align = ['left','center'],
        font = dict(color = 'Light Gray', size = 24),
        height = 40
    ),
    cells = dict(
        values = values,
        line_color = 'darkslategray',
        fill = dict(color = ['#ffb6c1', 'White']),
        font_size =  16,
        height = 40)
)
                         ]
                 )

fig.update_layout(width  =  4600, height  =  950,
                  title  =  'Table 1: Clustering Algorithms used for PUBG Hack Detection',
                  title_font  =  {'size': 24})
fig.update_xaxes(automargin = True)


## CSS stylesheet for formatting
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

## Instantiating the dashboard application
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets)


server = app.server
app.config['suppress_callback_exceptions'] = True

## Setting up the dashboard layout
app.layout = html.Div(
    [

### Inserting Logo into Heading and centering it
        html.Div(
            [
                html.Img(src = 'data:image/png;base64,{}'
                         .format(encoded_image1.decode())
                        )
            ],
            
            style = 
            {
                'display': 'flex', 'align-items': 'center',
                'justify-content': 'center'
            }
        ),

               
## Insert header for Problem
        html.Div(
            [
                html.H2("Problem" )
            ], 
        ),

# Insert Markdown for Problem / Background Information   
        html.Div(
            [
                dcc.Markdown(
                    ''' 
Player Segmentation is a relatively new application in video game analytics.  Being able to cluster certain groups to assess the efficiency of developers is a new avenue for maintaining and creating video games. Also, Player Segmentation can be used to identify players who are using a third-party software to gain an unfair advantage.
 
Now, you must ask why can't these development teams have some sort of detection inside their video games to solve this problem. However, they do! The issue is that when the developers update their anti-cheat detection, the hackers will update their cheats. With that in mind, there will be certain periods where the hackers will run rampant because the developers have not patched in their updated protocols.
 

Playerunknown's Battleground (PUBG) is a video game, which set the standard for preceding games in the Battle Royale genre. The main goal is to survive at all costs, as you are pitted against other human opponents in a large battlefield. PUBG addresses the downtime period with an in-game report system, where a player can observe another's player's actions through a replay and identify if there is unfair play. If enough people report an individual, they will receive a temporary ban from the platform and a staff member from PUBG will provide the final judgment.
 
I will identify a few problems with this method: 
* False-Reporting can lead to a waste of staff members time.
* There are only so many cases a staff member can oversee.
 
My approach is to see if we can use Machine Learning, specifically Unsupervised Learning to cluster player game data to address this hacker issue. 
                    '''
                )
            ], style = {'fontSize' : 20, 'font-family': 'Helvetica'}
        ),

              
## Insert header for Dataset
        html.Div(
            [
                html.H2("Dataset" )
            ], 
        ),

# Insert Markdown for Dataset  
        html.Div(
            [
                dcc.Markdown(
                    ''' 
The data was scraped off pubg.me, preprocessed and distributed through Kaggle. The dataset includes in-game player statistics relating to PUBG key performance indicators (KPIs).
'''              
                )
            ], style = {'fontSize' : 20, 'font-family': 'Helvetica'}
        ), 
        
## Insert header for Dataset
        html.Div(
            [
                html.H2("Feature Engineering" )
            ]
        ),

# Insert Markdown for Dataset  
        html.Div(
            [
                dcc.Markdown(
                    ''' 
The main objective of feature engineering was to target features, which indicates a player's solo performance without assistance from teammates. Next, with the knowledge that hackers use aim-assistance scripts (aids in placing the aiming cursor on opponent hitboxes), we'd expect certain features such as Kill-Death Ratio and Headshot-Kill Ratio to be directly affected. With that said, hackers are more likely to have greater Win Ratios and Top 10 Ratios because of these unfair advantages. For more information on the correlations between the features selected, check out the EDA Dashboards [1](https://pubg-eda-part1-dash.herokuapp.com/),[2](https://pubg-eda-part2-dash.herokuapp.com/). '''
                )
            ], style = {'fontSize' : 20}
        ),    
        
                
# Insert Header for Results
        html.Div(
            [
                html.H3("Results")
            ]
        ),
        
# Insert Markdown for Results        
        html.Div(
            [
                dcc.Markdown(
                    ''' 
As I began with the formulation of this problem in player segmentation, I started with implementing Lloyd's Algorithm (K-means Clustering) because there were various game analytic publications that presented promising results using K-means Clustering for Player Segmentation [[1](https://gameanalytics.com/blog/introducing-clustering-iv-case-tera-online.html)]. Beginning with parameter tuning using scoring metrics such as Inertia and Silhouette Score to identify the number of clusters, which resulted in two and four clusters as presented in Figures 1(a,b), respectively. 
                  '''
                ),
                
# Insert Optimal Cluster Plots                
                html.Div(
                    [
                        html.Div(
                            [
                                html.Img(src = 'data:image/png;base64,{}'
                                         .format(encoded_image2.decode())
                                        )
                            ], className = 'six columns',
                            style = 
                            {
                                'display': 'flex', 'align-items': 'center',
                                'justify-content': 'center'
                            }
                        ),
                        html.Div(
                            [
                                html.Img(src = 'data:image/png;base64,{}'
                                         .format(encoded_image3.decode())
                                        )
                            ], className = 'six columns',
                            style = 
                            {
                                'display': 'flex', 'align-items': 'center',
                                'justify-content': 'center'
                            }
                        ),
                    ], className = 'row'
                ) 
            ], style = {'fontSize' : 20, 'font-family': 'Helvetica'} 
        ),
        

# Insert Markdown for Results
        html.Div(
            [
                dcc.Markdown(
                    '''
After analyzing the clustering labels, I was skeptical about the high number of hackers present in our dataset. Despite the minimal amount of information on the internet regarding hackers in PUBG, I was able to get a factual statistic of 1.5 million accounts that were banned for hacking and at the time there were 26 million accounts created worldwide [[2](https://gamerant.com/playerunknowns-battlegrounds-cheater-ban-count/)]. Using that statistic, I calculated 5.8% of the population were hackers. And with such a minuscule amount of hackers, I decided to reformulate the problem as anomaly detection or outlier detection.
Several anomaly detection algorithms were selected based on their recent success in other applications: 
* Density-based spatial clustering of applications with noise (DBSCAN)
* Local Outlier Field (LOF)
* Elliptic Envelope (EE)
* Isolation Forest (IF)

Table 1 presents pertinent results for each clustering algorithm, but there was significant misclassification exhibited. And this misclassification was characterized by visual inspection of the clusters itself and examined various "outliers" labeled by these algorithms to verify based on my own domain experience. Finally, I selected Isolation Forest as my best performing algorithm.
                    '''
                ),
                
                html.Div(
                    [
                        dcc.Graph(figure = fig),
                    ],
                ),  
            ], style = {'fontSize' : 20, 'font-family': 'Helvetica'} 
        ),

        
# Insert Markdown for Results
        html.Div(
            [
                dcc.Markdown(
                    ''' 
After analyzing all the clustering algorithms, I researched dimensional reduction techniques to aid in minimizing the misclassification when clustering. Some algorithms have difficulty when dealing with the large number of dimensions (features) and dimensional reduction techniques minimize the number of features while keeping the essential information. Also, it may aid in finding new feature combinations to cluster the data. Therefore, Prinicipal Components Analyiss was experimented with all the algorithms used previously. Two iterations of PCA were utilized: All features and selected features (Kill-Death Ratio, Headshot-Kill Ratio, Win Ratio, and Top 10 Ratio). 

Before, we can apply PCA, we need to find the optimal number of components to explain at least 85% of the variance in the dataset as presented in Figures 2(a,b).

                  '''
                ),
                
                html.Div(
                    [
# Insert PCA Images
                        html.Div(
                            [
                                html.Img(src = 'data:image/png;base64,{}'
                                         .format(encoded_image4.decode())
                                        )
                            ], className = 'six columns',
                            style = 
                            {
                                'display': 'flex', 'align-items': 'center',
                                'justify-content': 'center'
                            }
                        ),
                        
                        html.Div(
                            [
                                html.Img(src = 'data:image/png;base64,{}'
                                         .format(encoded_image5.decode())
                                        )
                            ], className = 'six columns',
                            style = 
                            {
                                'display': 'flex', 'align-items': 'center',
                                'justify-content': 'center'
                            }
                        ),
                    ], className = 'row'
                ),
                dcc.Markdown(
                    ''' 
Unfortunately, both iterations of PCA in combination with all those algorithms performed previously produced identical and uninsightful results.    
                    '''
                ),
                
                html.Div(
                    [
                  #      dcc.Graph(figure = fig),
                    ],
                ),  
            ], style = {'fontSize' : 20, 'font-family': 'Helvetica'} 
        ),

        
## Insert header for Dataset
        html.Div(
            [
                html.H2("Closing Remarks / Future Work" )
            ]
        ),

# Insert Markdown for Dataset  
        html.Div(
            [
                dcc.Markdown(
                    ''' 
                    I am proud to present a [hacker detection tool](https://pubg-hacker-detection-app.herokuapp.com/) in which I developed using the clustering algorithms discussed previously. Caution, it does take a couple of minutes to load, but I hope you enjoy the tool. Now I would like to mention possible future modifications to the algorithms used and application developed.
                    
                    Algorithms:
                    * Keep up-to-date with recent research in anomaly detection methods.
                    * Optimize the parameters in the algorithms used.
                    * Explore different distance metrics in these algorithms.
                    * Explore additional dimensional reduction techniques besides PCA.
                    
                    Application:
                    * Scale the application to include master and slave protocol to have the slaves handle all the computations, while the master distributes the requests.
                    * Dynamically scrape the dataset from pubg.me to gain more insight.
                    '''              
                )
            ], style = {'fontSize' : 20,  'font-family': 'Helvetica'}
        ), 
    ],  style = {'font-family': 'Helvetica'}
)

    

if __name__ == '__main__':
    app.run_server(debug = True)