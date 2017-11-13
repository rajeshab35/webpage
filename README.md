# IRE Major Project
Multi-faceted trust in recommendation

# Github links
<a href="https://github.com/varunnaganathan/MTrust">mTrust: Discerning Multi-Faceted Trust in a Connected World</a> <br/>
<a href="https://github.com/rajeshab35/Circle-based-Recommendation">Circle-based Recommendation in Online Social Network</a> 
# Problem Statement
Improving Recommendation system accuracy by using category-specific social trust circles from social network data combined with available rating data.

# Introduction
## mTrust: Discerning Multi-Faceted Trust in a Connected World:
Trust between users and entities has often been assumed to be singular.The paper however challenged this concept by bringing in the multifaceted trust relations indicating multiple and heterogeneous relationships between users.
We aim to apply this concept to a product review system in order to reveal that trust between users is not homogenous but rather each user trusts other users in specific categories or “facets”.
Experimental results on real-world data from Epinions and Ciao show that our work of discerning multi-faceted trust can be applied to improve the performance of tasks such as rating prediction, facet-sensitive ranking, and status theory.

## Circle-based Recommendation in Online Social Network:
Online social network information promises to increase recommendation  accuracy  beyond  the capabilities  of  purely rating/feedback-driven  recommender  systems.As one of their major benefits, social network based  approaches  have  been  shown  to  reduce  the  problems  with cold start users.
 As to better serve users’ activities across different domains, many online social networks now support a new feature of “Friends Circles”,  which refines the domain-oblivious “Friends” concept.
In real life , a  user  may  trust  different  subsets  of friends regarding different domains.  Unfortunately, in most existing multi-category rating datasets, a user’s social connections from all categories are mixed together.
In this Project, we propose to implement circle-based recommendation system that  focus on inferring category-specific social trust circles from available rating  data  combined  with  social  network  data.We outline several variants of weighting friends within circles based on their inferred expertise levels.

# Dataset
Epinions is a consumer opinion website where users can review items and  assign numeric ratings. The trust values between users are binary. It  consists  of  ratings  from  71,002 users who rated a total of 104,356 different items from 451 categories.   The  total  number  of  ratings  is  571,235.
Dataset Url - http://alchemy.cs.washington.edu/data/epinions/

# Approach  
We first start with the basic matrix factorization model and then add one more factor that consider the social interaction information into the model for better personalization.
## Trust Circle Inference 
A user v is in the inferred circle of user u, i.e., in the set Cu(c),if and only if the following two conditions hold:
S(u,v) > 0 in the (original) social network, and Nu(c) > 0 and Nv(c) > 0 in the rating data,
where Nu(c) denotes  the  number  of  ratings  that  user u has assigned to items in category c. Otherwise, user v is not in the circle of u concerning category c.
### Trust Value Assignment, Equal Trust
Each user v in the inferred circle of user u gets assigned the same trust value.So S(c) matrix can be filled using following formula,
### Expertise-based Trust 
The goal is to assign a higher trust value or weight to the friends that are experts in the circle or category.As an approximation to their level of expertise, we use the numbers of ratings they assigned to items in the category.The idea is that an expert in a category may have rated more items in that category than users who are not experts in that category.
### Trust Splitting
Given that u trusts v,if v has more ratings in category c1 than in c2,it is more likely that u trusts v because of v’s ratings in c1 than v’s ratings in c2.Now if u and v simultaneously have ratings in multiple categories,the trust value of u towards v should be split across those commonly rated categories


# Results
RMSE comparison for MTrust for variable alpha values:

| Alpha value | RMSE(our implementation) | RMSE (as per paper)                           |
|-------------|--------------------------|-----------------------------------------------|
| 0.01        | 3.812217197              | Not mentioned(but supposed to give max error) |
| 0.3         | 1.1953                   | 0.98675                                       |
| 0.8         | 1.259971543              | 1.0423                                        |

Circle-based Recommendation in Online Social Networks,Below results shows RMSE error for first 1000 users, equal trust division gave the least RMSE error, while expert based trust gave better result than trust split.
Note:- The difference between published result and ours is evident as it was run on 71,002 users. 1000 user restriction is because of our memory and computation limitations. 

| category                 | equal trust | expert based | Trust split |
|--------------------------|-------------|--------------|-------------|
| Cars                     | 0.9816      | 1.084        | 1.163       |
| Online_Stores_&_Services | 0.8973      | 0.8787       | 0.9206      |
| Videos_&_DVDs            | 0.5951      | 0.5914       | 0.6201      |
| Books                    | 0.7148      | 0.729        | 0.7703      |
| Music                    | 0.6481      | 0.7297       | 0.7853      |
| Video_Games              | 0.8505      | 0.7898       | 0.8877      |
| toy                      | 0.7603      | 0.7726       | 0.8294      |
| Destinations             | 0.7725      | 0.8404       | 0.9231      |
| software                 | 0.9382      | 0.8978       | 0.9649      |
| Avg RMSE error           | 0.79537778  | 0.8126       | 0.873822    |

# Implementation detail
circle based recommendation

## 1. loading data form files to panda data frame
Products.csv -> Product_df (stores all product data)<br/>
User_Trust_Network.csv -> User_Trust_Network_df (stores trustee list of all users)<br/>
Rating.csv -> rating_df (stores all rating information)


### 1.1 funcation/variable declaration


```python
import sys
import os
import subprocess
import pandas as pd
import re
import numpy as np
from collections import defaultdict

np.random.seed(0)
d = 10
Dict_Product_Category={}
No_of_items_in_category={}
TRAINING_MATRICES={}
Category_Circle = defaultdict(lambda : defaultdict(float))
Category_to_List_of_items={}
user_rating_dict = {}
cat_list = ['Books', 'Music','Video Games','Toys','Software','Destinations','Cars']

#helper function for creating trustee set
def Trustee_List(Trustee):
    Trustee_set=set()
    for trustee in Trustee.split(","):
        Trustee_set.add(int(trustee))
    return Trustee_set

#helper function to store product category information in "Category_to_List_of_items"
def Load_Dictionary(Product_df):
    for i in range(len(Product_df)):
        product_id=Product_df['Product_id'][i]
        Category_name=Product_df['Category_names'][i]
        Dict_Product_Category[product_id]=Category_name
        if Category_name not in No_of_items_in_category:
            No_of_items_in_category[Category_name]=0
        if Category_name not in Category_to_List_of_items:
            Category_to_List_of_items[Category_name]=list()
        No_of_items_in_category[Category_name]+=1
        Category_to_List_of_items[Category_name].append(product_id)
```

### 1.2 load files


```python
Product_df = pd.read_csv("Products.csv") 
Load_Dictionary(Product_df)

User_Trust_Network_df = pd.read_csv("User_Trust_Network.csv", sep=':')  
User_Trust_Network_df['Trustee'] = map(Trustee_List,User_Trust_Network_df['Trustee'])

items=len(Product_df)

rating_df=pd.read_csv("Rating.csv")
List_Users=list(set(rating_df['User_id']))
users=len(List_Users)

```

### 1.3 adding column "category" and "test flag" to rating data structure "rating_df"


```python
for userid in List_Users:
    user_rating_dict[userid] = {}
    user_rating_dict[userid]['ratings'] = {}

List_Category_row = []
for i in range(len(rating_df)):
    List_Category_row.append(Dict_Product_Category[rating_df['Product_id'][i]])
    userid = rating_df['User_id'][i]
    # default all as flase
    # flase -> training data
    # true  -> test data
    user_rating_dict[userid]['ratings'][i] = False

rating_df['Category'] = List_Category_row

# adding test range and test len 
for userid in List_Users:
    user_rating_dict[userid]['test_start'] = 0
    user_rating_dict[userid]['rating_len'] = len(user_rating_dict[userid]['ratings'])
    user_rating_dict[userid]['test_len'] = int(0.2 * user_rating_dict[userid]['rating_len'])
    user_rating_dict[userid]['test_end'] = user_rating_dict[userid]['test_len']
    for row_no,row in enumerate(user_rating_dict[userid]['ratings'].keys()):
        if row_no >= user_rating_dict[userid]['test_start'] and row_no <  user_rating_dict[userid]['test_end']:
            user_rating_dict[userid]['ratings'][row] = True
    
# adding test column to training_df
test_flag_row = []
for i in range(len(rating_df)):
    userid = rating_df['User_id'][i]
    test_flag_row.append(user_rating_dict[userid]['ratings'][i])

rating_df['test_flag'] = test_flag_row

```

# 2. Training phase

## 2.1 setting up training parameters


```python
#traning parameters
lamda = 0.1      # regularization constant
l = 0.001       # learning rate
beta = 20        # Social information weight  
iterations = 100  #change this

# Error derivative wrt Q
def Error_derivative_wrt_Q(I, R_Predicted,R, P,Q, S):
    First_term = np.matmul(np.multiply(I,(R_Predicted-R)), P)
    Second_term = lamda * Q
    Third_term = beta * np.subtract(Q, np.matmul(S,Q))
    fourth_term = beta *  np.matmul(S.transpose(),np.subtract(Q, np.matmul(S,Q))) 
    return First_term + Second_term + Third_term - fourth_term 

# Error derivative wrt P
def Error_derivative_wrt_P(I,R_Predicted,R,P,Q):
    First_term = np.matmul(np.multiply(I,(R_Predicted-R)).transpose(), Q)
    Second_term = lamda * P
    return First_term + Second_term 

# Training 
def Training(I,R_Predicted,R,P,Q,S):
    First_term = np.sum(np.sum(np.multiply(R - np.multiply(R_Predicted,I), R - np.multiply(R_Predicted,I)))) / (2.0)
    Second_term = (lamda/(2.0)) *(np.linalg.norm(Q,ord='fro') + np.linalg.norm(P,ord='fro'))
    Third_term_temp = np.subtract(Q, np.matmul(S,Q))
    Third_term = (beta/(2.0)) * pow((np.linalg.norm(Third_term_temp,ord='fro')),2)
    return First_term + Second_term + Third_term

```

## 2.2 create training datastructure and matrices


```python
# creating test and training rating set
Training_df = rating_df[ rating_df.test_flag == False ]
Testing_df = rating_df[ rating_df.test_flag == True ]

# for each category create training matrices P,Q,R,S,r,I and add it to dictionary TRAINING_MATRICES
with open("Category.txt") as Category_file:
for Category in Category_file:
    if Category.strip() not in cat_list:
        continue
    if Category.strip() in No_of_items_in_category:
        #item matrix
        P = np.random.randn(No_of_items_in_category[Category.strip()], d)
        P = P.astype(dtype = np.float16)

        #user matrix
        Q = np.random.randn(users, d)
        Q = Q.astype(dtype = np.float16)

        #existing rating matrix users * items 
        R = np.empty([users, No_of_items_in_category[Category.strip()]], dtype = np.float16)
        R.fill(0.0)

        #user trust matrix, users * users
        S = np.empty([users,users], dtype = np.float16)
        S.fill(0.0)

        #bias matrix
        r = np.empty([users,No_of_items_in_category[Category.strip()]], dtype = np.float16)       
        r.fill(0.0)

        I = np.empty([users, No_of_items_in_category[Category.strip()]], dtype = np.float16)
        I.fill(0.0)

        TRAINING_MATRICES[Category.strip()]=[P,Q,R,S,r,I]
```

## 2.3 initialize training matrices


```python
# Filling matrix r
Rating_count={}
r_values={}
for category,rating_group in Training_df.groupby('Category'):
    if category not in cat_list:
        continue
    r_values[category]= np.mean(rating_group['Rating'])
    Rating_count[category]=len(rating_group['Rating'])

for category in r_values:
    if category not in cat_list:
        continue
    r=TRAINING_MATRICES[category][4]
    r.fill(r_values[category])

# Filling matrix R and I
for user,item,category,rating in zip(Training_df['User_id'],Training_df['Product_id'],Training_df['Category'],Training_df['Rating']):
    if category not in cat_list:
        continue
    if category in TRAINING_MATRICES:
        R = TRAINING_MATRICES[category][2]
        I = TRAINING_MATRICES[category][5]
        row_index = List_Users.index(user)
        col_index = Category_to_List_of_items[category].index(item)
        R[row_index][col_index] = rating
        I[row_index][col_index] = 1

# Filling Matrix S as Category_Circle (Equal trust)
for user, rating_group in Training_df.groupby('User_id'):
    if user in List_Users:
        for rating_entry in rating_group['Product_id']:
                Category_Circle[Dict_Product_Category[rating_entry]][user] += 1

for Category in Category_Circle.keys():
    if category not in cat_list:
        continue
    category_user_list = set(Category_Circle[Category].keys())
    for i in range(users):
        user = User_Trust_Network_df['User_id'][i]
        if user in category_user_list:
            trustee = User_Trust_Network_df['Trustee'][i]
            Same_Category_trustee = trustee.intersection(category_user_list)
            S = TRAINING_MATRICES[Category][3]

            val = 1.0 / float(len(Same_Category_trustee))
            if total_row_WT != 0:
                for trustee_user in Same_Category_trustee:
                    row_index = List_Users.index(user)
                    col_index = List_Users.index(trustee_user)
                    S[row_index][col_index] = val
```

## 2.4  train
for each category calculate RMSE value and update matrix P(product profile matix) and Q(user profile matix)


```python
Old_Error=1000
for category in Category_Circle.keys():
    if category not in cat_list:
        continue
    P=TRAINING_MATRICES[category][0]
    Q=TRAINING_MATRICES[category][1]
    R=TRAINING_MATRICES[category][2]
    S=TRAINING_MATRICES[category][3]
    r=TRAINING_MATRICES[category][4]
    I=TRAINING_MATRICES[category][5]


    R_Predicted = r + np.matmul(Q, P.transpose())   
    # print "R_Predicted : ",R_Predicted.shape
    ofd.write("################################"+"\n")
    ofd.write(category+"\n")

    for itr in range(iterations): 

#         print "iteration no : ", str(i+1)

        error_der_P = Error_derivative_wrt_P(I,R_Predicted,R,P,Q)
        error_der_Q = Error_derivative_wrt_Q(I,R_Predicted,R,P,Q,S)  

        # P = np.subtract(P, np.multiply(l, error_der_P))  
        for i in range(No_of_items_in_category[category]):
            for j in range(d):
                P[i][j] -= (l * error_der_P[i][j])

        Q = np.subtract(Q, np.multiply(l, error_der_Q))

        R_Predicted = r + np.matmul(Q, P.transpose())

        RMSE_train = np.sqrt(np.sum(np.square(np.subtract(np.multiply(I,R_Predicted),R)))/float(Rating_count[category]))

        New_Error = Training(I,R_Predicted,R,P,Q,S)

        print itr,": Test round: ",loop_counter," ,", category,' RMSE_train : ', str(RMSE_train)
        # print "P[0]: ", P[0]
        # print "Q[0]: ", Q[0]
        ofd.write("iteration : "+str(itr)+" # Test round : "+str(loop_counter)+" # RMSE_train : "+str(RMSE_train)+"\n")

        if itr>0:    
            if ((Old_Error-New_Error)<10):
                break    
        Old_Error = New_Error
```

# 3. test
for each category use trained matrix P and Q to predict ratings and caluculate RMSE value


```python
for category,rating_group in Testing_df.groupby('Category'):
    if category not in cat_list:
        continue
    if len(rating_group) == 0:
        continue
    if category in TRAINING_MATRICES:
        P=TRAINING_MATRICES[category][0]
        Q=TRAINING_MATRICES[category][1]
        R=TRAINING_MATRICES[category][2]
        S=TRAINING_MATRICES[category][3]
        r=TRAINING_MATRICES[category][4]

        R_Test=np.empty(R.shape, dtype = np.float16)
        I_Test=np.empty(R.shape, dtype = np.float16)
        R_Test.fill(0.0)
        I_Test.fill(0.0)
        for user,item,rating in zip(rating_group['User_id'],rating_group['Product_id'],rating_group['Rating']):
            row_index=List_Users.index(user)
            col_index=Category_to_List_of_items[category].index(item)
            R_Test[row_index][col_index]=rating
            I_Test[row_index][col_index]=1
        
        # predict ratings
        R_Predicted = r + np.matmul(Q, P.transpose())

        RMSE_test = np.sqrt(np.sum(np.square(np.subtract(np.multiply(I_Test,R_Predicted),R_Test)))/float(len(rating_group)))
        #print I_Test #,R_Predicted)
        #print np.sum(np.square(np.subtract(np.multiply(I_Test,R_Predicted),R_Test)))
        print category, '  RMSE_test : ', str(RMSE_test)
        ofd.write(category+" # RMSE_test : "+str(RMSE_test)+"\n")
        cat_error_val[category] += RMSE_test
