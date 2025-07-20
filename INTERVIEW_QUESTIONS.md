# ML Jobs and Interview Questions 2025 💹 🐱‍💻
🔥 Top companies to work for in AI and 100+ ML interview questions open sourced from [neuraprep.com](https://neuraprep.com)

🙏 Feel free to submit a new job posting or suggest a change at team@neuraprep.com

🎓 On Hedge Funds side, only those that hire on ML actively and continuously were included

------

Top companies hiring on ML (subjectively ranked based on perception, culture, program, prestige and pay): 

1️⃣ Meta - OpenAI - Anthropic - Nvidia. 

2️⃣ Citadel (Securities) - Netflix - Google - TwoSigma. 

3️⃣ RunwayML - Uber - xAI. 

4️⃣ Microsoft - Tesla - Tiktok - Stripe - Cruise. 

5️⃣ Lambda - Figure AI - Scale - Coinbase - Reddit - Adobe.

⚠️ Disclaimer: this is just a very rough ranking based on highly subjective opinions. In order for a company to make it to this list, they have to pay at least $300k/yr in average total compensation for ML roles. Tier 1 and 2 need to pay over $500k/yr.


------

### ML Interview Questions

1. **[Startup] Learning Rate Significance**  
  Why do we take smaller values of the learning rate during the model training process instead of bigger learning rates like 1 or 2?

2. **[Startup] Train-Test Split Ratio**  
  Is it always necessary to use an 80:20 ratio for the train test split? If not, how would you decide on a split?

3. **Covariance vs Correlation**  
  What is the difference between covariance and correlation?

4. **Skewed Distributions Tendencies**  
  What happens to the mean, median, and mode when your data distribution is right skewed and left skewed?

5. **[Amazon] Robustness to Outliers**  
  Which one from the following is more robust to outliers: MAE or MSE or RMSE?

6. **[Automattic] Content vs Collaborative Filtering**  
  What is the difference between the content-based and collaborative filtering algorithms of recommendation systems?

7. **[TripAdvisor] Restaurant Recommendation System**  
  How would you build a restaurant recommendation for TripAdvisor?

8. **[Stanford] Ensemble Model Performance**  
  Why do ensembles typically have higher scores than the individual models? Can an ensemble be worse than one of the constituents? Give a concrete example.

9. **[Bosch] Focal Loss in Object Detection**  
  Elaborate on the focal loss and its application in object detection.

10. **[Hedge Fund] Clock Hands Angle**  
  What is the angle between the hands of a clock when the time is 3:15?

11. **[Startup] Optimizing Labeled Data**  
  Getting labeled data in real world applications is not cheap, how do you optimize the number of labeled data? Give 3 popular strategies used in the industry to solve this problem.

12. **Few-Shot Learning Steps**  
  What steps does few-shot learning (sometimes grouped with meta learning) involve?

13. **[Startup] Greedy Layer-wise Pretraining 1**  
  What is greedy layer-wise pretraining? How does it compare to freezing transfer learning layers?

14. **Freezing Transformer Layers**  
  Why might you want to freeze transfer learning layers in the context of transformers?

15. **Dropout During Inference**  
  What happens to dropout during inference? If at the training stage we randomly deactivate neurons, then do we do the same when predicting?

16. **[Tiktok] Importance of Variation in VAEs**  
  Why do we need 'variation' in the variational autoencoder, what would happen if we remove the 'variation'? Explain how this relates to the difference between NLU and Natural Language Generation.

17. **Generative Model: Training vs Inference**  
  How does a generative model differ during training and inference in the context of text generation?

18. **Subword Tokenization Explanation**  
  What is subword tokenization, and why is it preferable to word tokenization? Name a situation when it is NOT preferable.

19. **Use of Sigmoid for Numerical Prediction**  
  Suppose you want to build a model that predicts a numerical quantity such as loan amount, investment amount, product price, etc. Why might you feed the final layer through a sigmoid function?

20. **[Hedge Fund] Function Derivative Zero Sum**  
  What function yields 0 when added to its own derivative?

21. **Continuous Binary State Function**  
  In a binary state, there are only two possible values: 0 or 1, which can represent off/on, false/true, or any two distinct states without any intermediate values. However, in many computational and real-world scenarios, we often need a way to express not just the two extreme states but also a spectrum of possibilities between them. Give an example of a function that represents a continuous version of a binary state (bit) and explain why.

22. **[Circle K] PCA and Correlated Variables**  
  You are given a dataset. The dataset contains many variables, some of which are highly correlated and you know about it. Your manager has asked you to run a PCA. Would you remove correlated variables and why?

23. **Dot Product Complexity**  
  How does the dot product of two vectors scale with N?

24. **Deep Learning Success**  
  Though the fundamentals of Neural nets were known since the 80s, how does this explain the success of Deep Learning in recent times?

25. **[Startup] Model Convergence Evaluation**  
  You are training a neural network and you observe that training and testing accuracy converges to about the same. The train and testset are built well. Is this a success? What would you do to improve the model?

26. **[Facebook] Unfair Coin Probability**  
  There is a fair coin (one side heads, one side tails) and an unfair coin (both sides tails). You pick one at random, flip it 5 times, and observe that it comes up as tails all five times. What is the chance that you are flipping the unfair coin?

27. **[Quora] Drawing normally**  
  You are drawing from a normally distributed random variable X ~ N(0, 1) once a day. What is the approximate expected number of days until you get a value of more than 2

28. **[Airbnb] Fair Odds from Unfair Coin**  
  Say you are given an unfair coin, with an unknown bias towards heads or tails. How can you generate fair odds using this coin?

29. **[Airbnb] Customer Churn MLE**  
  Say you model the lifetime for a set of customers using an exponential distribution with parameter l, and you have the lifetime history (in months) of n customers. What is the Maximum Likelihood Estimator (MLE) for l?

30. **[Lyft] Probability of Feature Shipping**  
  Say that you are pushing a new feature X out. You have 1000 users and each user is either a fan or not a fan of X, at random. There are 50 users out of 1000 that do not like X. You will decide whether to ship the feature or not based on sampling 5 distinct users independently and if they all like the feature, you will ship it. What's the probability you ship the feature? How does the approach change if instead of 50 users , we have N users who do not like the feature, how would we get the maximum value of unhappy people to still ship the feature? -

31. **[Hedge Fund] Probability of Ruin in Gambling**  
  I have $50 and Im gambling on a series of coin flips. For each head I win $2 and for each tail I lose $1. What's the probability that I will run out of money? - .

32. **[Hedge Fund] Regression Slope Inversion**  
  Suppose that X and Y are mean zero, unit variance random variables. If least squares regression (without intercept) of Y against X gives a slope of b (i.e. it minimizes [(Y - bX)^2 ]), what is the slope of the regression of X against Y?

33. **[SimPrints] Face Verification System Steps**  
  What do you think are the main steps for a face verification system powered by ML? Would CNN work well as a model?

34. **Mini-Max Optimization**  
  Which method is used for optimizing a mini-max based solution?

35. **[Microsoft] Handling Missing Data**  
  You are given a dataset consisting of variables having more than 30% missing values. Let's say out of 50 variables, 8 have more than 30% missing values. How do you deal with them?

36. **Type I vs Type II Errors**  
  What is the difference between Type I and Type II errors? Follow up: better to have too many type I or type II errors in a solution?

37. **[Scaleai] Multimodal Attention in LLMs**  
  How do you complete the alignment of different modal information in a multimodal Large Language Model? How does the attention mechanism still work with cross-modal inputs?

38. **[SentinelOne] RNN vs Transformer**  
  How do RNNs differ from transformers? Mention 1 similarity and 2 differences.

39. **[Stanford] SGD Loss Function Behavior**  
  Is it necessary that SGD will always result in decrease of loss function?

40. **Approximate Solutions in Training**  
  Why might it be fine to get an approximate solution to an optimization problem during the training stage?

41. **Backpropagation Computational Cost**  
  What operation is the most computational expensive in backpropagation and why?

42. **Noisy Label Classification**  
  How would you do classification with noisy labels or many incorrect labels?

43. **Logistic Regression on Linearly Separable Data**  
  What would happen if you try to fit logistic regression to a perfectly linearly separable binary classification dataset. What would you do if given this situation, assuming you must preserve logistic regression as the model?

44. **[Faang] Model Production Issue**  
  Your training, validation and test accuracy are more than 90% accuracy. Once in production, the model starts to behave weirdly. How will you identify what's happening and how will you correct it?

45. **[Faang] Self-attention Time Complexity**  
  What is the time complexity of the Self-attention layer?

46. **'Random' in Random Forest**  
  What is random in a random forest, are there any benefits for this randomness?

47. **Gradient Descent vs Analytical Solution**  
  Why do we need gradient descent instead of just taking the minimum of the N dimensional surface that is the loss function?

48. **Hessian Matrix in Optimization**  
  What is the role of the Hessian matrix in optimization, and why is it not commonly used in training deep neural networks?

49. **Combining Normalized Features**  
  If two features are embedding outputs - dimensions 1xN, 1xM - and one feature is single value output - 1x1 - and all feature values are normalized to between -1 and 1, how can these be combined to create a classification or regression output?

50. **A/B Test Analysis**  
  A company runs an A/B test for a donation group but the conversion didn't increase / statistically-significant increase what you do?

51. **[Meta] Impact Analysis of New User**  
  What kind of analysis will you run to measure the effect of a facebook user when their younger cousin joins?

52. **Logistic Regression vs Decision Tree**  
  When would you use logistic regression over a decision tree? Which one would you use when the classification problem deals with perfectly linearly separable data?

53. **Improving Model Prediction**  
  You have a model with a high number of predictors but poor prediction power. What would you do in this case?

54. **Naive Bayes with Laplace Smoothing**  
  When should we use Naive Bayes with Laplace smoothing? Give a practical example

55. **Activation Function Comparison**  
  Which of the following activations has the highest output for x=2: Tanh, ReLu, Sigmoid, ELU? Without computing the functions, provide an explanation.

56. **ReLU Activation Issue**  
  You are training a model using ReLu activations functions. After some training, you notice that many units never activate. What are some plausible actions you could take to get more units to activate?

57. **Variance of Duplicated Data**  
  What would happen to the variance of whole data if the whole data is duplicated?

58. **[Google] Identify Synonyms from Corpus**  
  Say you are given a large corpus of words. How would you identify synonyms? Mention several methods in a short answer.

59. **[Airbnb] Model Airbnb Revenue**  
  Say you are modeling the yearly revenue of new listings of airbnb rentals. What kinds of features would you use? What data processing steps need to be taken, and what kind of model would run? Would a neural network work?

60. **[Google] Linear Regression with Noise**  
  Say we are running a linear regression which does a good job modeling the underlying relationship between some y and x. Now assume all inputs have some noise added, which is independent of the training data. What is the new objective function and effects on it?

61. **[Netflix] Gradient Descent in K-Means**  
  If you had to choose, would you use stochastic gradient descent or batch gradient descent in k-means? Does k-means use any gradient descent to optimize the weights in practice?

62. **[Netflix] EM Algorithm Use Cases**  
  When is Expectation-Maximization useful? Give a few examples.

63. **Dependence vs Correlation**  
  What's the difference between dependence and correlation?

64. **[BioRender] Transformers in Computer Vision**  
  How can transformers be used for tasks other than natural language processing, such as computer vision (ViT)?

65. **Convert NN Classification to Regression**  
  How would you change a pre-trained neural network from classification to regression?

66. **Train Loss Stagnation**  
  When it comes to training a neural network, what could be the reasons for the train loss not decreasing in a few epochs?

67. **High Momentum in SGD**  
  What might happen if you set the momentum hyperparameter too close to 1 (e.g., 0.9999) when using an SGD optimizer?

68. **Law of Large Numbers**  
  What is the Law of Large Numbers in statistics and how can it be used in data science?

69. **Selection Bias**  
  What is the meaning of selection bias and how to avoid it?

70. **Weight Decay Scaling Factor**  
  Why do we need a scaling factor in weight decay? Is it independent from batch size or learning rate?

71. **Fuzzy Logic Explanation**  
  What is fuzzy logic?

72. **Latent Variables in Stable Diffusion**  
  Why do we call the hidden states "latent variables" instead of embeddings in stable diffusion?

73. **[Robinhood] User Churn Prediction Model**  
  Walk me through how you'd build a model to predict whether a particular Robinhood user will churn.

74. **Ensemble Logistic Regression as Network**  
  Consider a binary classification problem and N distinct logistic regression models. You decide to take a weighted ensemble of these to make your prediction. Can you express the ensemble in terms of an artificial network? How?

75. **Feature Selection with Mutual Information**  
  Consider learning a classifier in a situation with 1000 features total. 50 of them are truly informative about class. Another 50 features are direct copies of the first 50 features. The final 900 features are not informative. Assume there is enough data to reliably assess how useful features are, and the feature selection methods are using good thresholds. How many features will be selected by mutual information filtering?

76. **Optimal Polynomial Degree for Regression**  
  We are trying to learn regression parameters for a dataset which we know was generated from a polynomial of a certain degree, but we do not know what this degree is. Assume the data was actually generated from a polynomial of degree 5 with some added Gaussian noise. For training we have 1000  pairs and for testing we are using an additional set of 100  pairs. Since we do not know the degree of the polynomial we learn two models from the data. Model A learns parameters for a polynomial of degree 4 and model B learns parameters for a polynomial of degree 6. Which of these two models is likely to fit the test data better?

77. **Softmax and Scaling**  
  For an n-dimensional vector y, the softmax of y will be the same as the softmax of c * y, where c is any non-zero real number since softmax normalizes the predictions to yield a probability distribution. Am I correct in this statement?

78. **Evaluation Metric for Criminal Identification**  
  You are hired by LAPD as a machine learning expert, and they require you to identify criminals, given their data. Since being imprisoned is a very severe punishment, it is very important for your deep learning system to not incorrectly identify the criminals, and simultaneously ensure that your city is as safe as possible. What evaluation metric would you choose and why?

79. **Batch Size and Minima**  
  Is it always a good strategy to train with large batch sizes? How is this related to flat and sharp minima?

80. **Logistic Regression on Synthetic Data**  
  You are building a classification model to distinguish between labels from a synthetically generated dataset. Half of the training data is generated from N(2,2) and half of it is generated from N(0,3). As a baseline, you decide to use a logistic regression model to fit the data. Since the data is synthesized easily, you can assume you have infinitely many samples. Can your logistic regression model achieve 100% training accuracy?

81. **ReLU before Sigmoid Issue**  
  You decide to use ReLU as your hidden layer activation, and also insert a ReLU before the sigmoid activation such that ^y = s(ReLU(z)), where z is the preactivation value for the output layer. What problem are you going to encounter?

82. **Handling Class Imbalance in Medical Imaging**  
  You're asked to build an algorithm estimating the risk of premature birth for pregnant women using ultrasound images. You have 500 examples in total, of which only 175 were examples of preterm births (positive examples, label = 1). To compensate for this class imbalance, you decide to duplicate all of the positive examples, and then split the data into train, validation and test sets. Explain what is a problem with this approach.

83. **[Stanford] Model-based Car Optimization**  
  Suppose you have built a model to predict a car's fuel performance (e.g. how many miles per gallon) based on engine size, car weight, etc. . . (e.g. many attributes about the car). Your boss now has the great idea of using your trained model to build a car that has the best possible fuel performance. The way this is done will be by varying the parameters of the car, e.g. weight and engine size and then using your model to predict fuel performance. The parameters will then be chosen such that the predicted fuel performance is the best. Is this a good idea? Why? Why not?

84. **Improving High Training Loss**  
  You want to solve a classification task with a neural network. You first train your network on 20 samples. Training converges, but the training loss is very high. You then decide to train this network on 10,000 examples. Is your approach to fixing the problem correct? If yes, explain the most likely results of training with 10,000 examples. If not, give a solution to this problem.

85. **CNN vs. Fully-connected for Images**  
  Alice recommends the use of convolutional neural networks instead of fully-connected networks for image recognition tasks since convolutions can capture the spatial relationship between nearby image pixels. Bob points out that fully-connected layers can capture spatial information since each neuron is connected to all of the neurons in the previous layer. Both are correct, but describe two reasons we should prefer Alice's approach to Bob's.

86. **Weight Initialization in Neural Networks**  
  You try a 4-layer neural network in a binary classification problem. You initialize all weights to 0.5. Is this a good idea? Briefly explain why or why not?

87. **Impact of Weight Sharing**  
  Does weight sharing increase the bias or the variance of a model? Why?

88. **PDF value range**  
  A probability density function (PDF) cannot be less than 0 or bigger than 1. Is that true? Why or why not?

89. **KNN Bias-Variance Tradeoff**  
  How does the bias-variance tradeoff play out for the k nearest neighbor algorithm as we increase k?

90. **RAG Explanation**  
  What is Retrieval-Augmented Generation (RAG)?

91. **RAG Limitations**  
  What are some limitations of RAG?

92. **LLM Tuning Techniques**  
  In the world of LLMs, choosing between fine-tuning, Parameter-Efficient Fine-Tuning (PEFT), prompt engineering, and retrieval-augmented generation (RAG) depends on the specific needs and constraints of your application. Explain what each one of these does.

93. **Biases in CNN**  
  You are building the next SOTA CNN for vision tasks following the architecture: (Layer input) => (Conv Layer) => (Batch Norm) => (Activation) => (Next Layer Input). The Conv layer has a set of learnable weights and biases but you decide not to train biases (hardcode them all to 0). Would the performance be affected compared to the same system with biases learning turned on?

94. **Transformer Encoder vs Decoder**  
  There are 2 deeper technical differences between the encoder and decoder in a transformer. Can you mention them?

95. **[Scaleai] Decoder vs Encoder Popularity**  
  Decoder only models have become much more popular recently compared to encoder models. The majority of NLU models are decoder only, why is that? Think about the advantage that encoder models have over decoder, why didn't that matter?

96. **Encoder-Decoder vs Decoder-Only**  
  Why do we need encoder-decoder models while decoder-only models can do everything?

97. **[Scaleai] GPT-4 Architecture and Training**  
  Can you describe the process by which GPT-4 generates coherent and contextually relevant text and why a decoder-only architecture was chosen? What input/output was it used for training?

98. **T5 vs GPT-4 for LLM**  
  A T5 or FlanT5 model is considered one of the best encoder-decoder models out there (as of 2024). In technical details, why aren't people using it at scale to train a large LLM that can compete with GPT4?

99. **[Spotify] Clustering Performance with Labels**  
  Can you suggest some ways in which the performance of a clustering algorithm can be measured when the labels are given?

100. **[Uber] Simulating Fair Die Roll**  
  How would you simulate the roll of a fair six-sided die using U(0,1) (uniform distribution) random number generator? How would you validate that the rolls are indeed fair?

101. **[Palo Alto Networks] Batch vs Instance Normalization Differences**  
  What are the differences between batch normalisation and instance normalisation? Give an example where the instance norm would be preferred.

102. **Bias-Variance Tradeoff**  
  Explain the concept of bias-variance tradeoff and its significance in machine learning.

103. **Curse of Dimensionality**  
  What is the curse of dimensionality and how does it affect machine learning algorithms?

104. **[Kayzen] Gradient Boosting Explanation**  
  How does gradient boosting work and why is it effective?

105. **[SYZYGY] Confusion Matrix Explanation**  
  Describe the concept of a confusion matrix and its components.

106. **Backpropagation Explanation**  
  How does backpropagation work in neural networks?

107. **Transfer Learning in Deep Learning**  
  What is transfer learning and how is it applied in deep learning?

108. **[JPMorgan] Bagging vs Boosting**  
  What is the difference between bagging and boosting in ensemble learning?

109. **[Apple] Feature Engineering with XGBoost**  
  Explain the concept of feature engineering and its importance in the context of tabular data with XGBoost as a predictive modeling tool.

110. **Activation Functions in Neural Networks**  
  What is the role of activation functions in neural networks?

111. **[Apple] ROC Curve Concept**  
  Describe the concept of an ROC curve and how it is used.

112. **Binary Classification Metrics**  
  What are the common metrics used for evaluating binary classification models?

113. **Feature Selection**  
  How do you perform feature selection in machine learning?

114. **Clustering Performance Metrics**  
  How do you measure the performance of a clustering algorithm?

115. **[Meta] Challenges with Large Datasets**  
  What are the challenges of working with large-scale datasets?

116. **Softmax in Multi-Class Classification**  
  How does the softmax function work in multi-class classification problems?

117. **Decision Trees Data Handling**  
  Explain how decision trees handle categorical and numerical data.

118. **[Amazon] Handling Outliers**  
  How do you handle outliers in a dataset?

119. **ML Deployment Challenges**  
  What are the challenges in deploying machine learning models to production?

120. **Feature Scaling Importance**  
  How do you perform feature scaling and why is it important?

121. **Linear Regression Coefficients**  
  How do you interpret the coefficients of a linear regression model?

122. **Tree-based Algorithms Advantages**  
  What are the advantages of using tree-based algorithms in machine learning? Mention as many as you know.

123. **Data Augmentation in Deep Learning**  
  Explain the concept of data augmentation and its importance in training deep learning models.

124. **Early Stopping in Neural Networks**  
  What is the role of early stopping in training neural networks?

125. **Time Series Cross-Validation**  
  How do you implement cross-validation for time series data?

126. **Homoscedasticity vs Heteroscedasticity**  
  Describe the difference between homoscedasticity and heteroscedasticity in regression analysis.

127. **[detikcom] Embedding Layer Purpose**  
  What is the purpose of using an embedding layer in neural networks?

128. **Batch Normalization in Neural Networks**  
  How does the batch normalization technique work in neural networks?

129. **Collaborative Filtering in Recommenders**  
  How does the collaborative filtering algorithm work in recommendation systems?

130. **Bag-of-Words Explanation**  
  Explain the concept of bag-of-words in natural language processing.

131. **Feature Selection Pitfalls**  
  What are the common pitfalls in feature selection?

132. **Gradient Boosting Overfitting**  
  How does the gradient boosting algorithm handle overfitting?

133. **Model Stacking Implementation**  
  How do you implement model stacking in ensemble learning?

134. **Random Forest Missing Values**  
  How does the random forest algorithm handle missing values?

135. **Training with Noisy Data**  
  What are the challenges in training machine learning models with noisy data?

136. **Data Augmentation in CV**  
  How do you perform data augmentation in computer vision tasks?

137. **Interpreting Hierarchical Clustering**  
  How do you interpret the results of a hierarchical clustering algorithm?

138. **CNN Feature Extraction**  
  How do you perform feature extraction using convolutional neural networks (CNNs)? -

139. **[Startup] Challenges of Limited Data**  
  What are the challenges in training machine learning models with limited data?

140. **GMM Clustering**  
  How does the Gaussian Mixture Model (GMM) perform clustering?

141. **PDP Model Interpretation**  
  Describe the process of model interpretation using partial dependence plots (PDP).

142. **Feature Selection with Info Gain**  
  How do you implement feature selection using information gain?

143. **PPO in Continuous Spaces**  
  How does the reinforcement learning algorithm PPO optimize policies for continuous action spaces?

144. **[Imubit] Optimal Clusters in K-Means**  
  How does the k-means clustering algorithm determine the optimal number of clusters?

145. **[Meta] Large-scale Image Classification System**  
  Architect a large-scale image classification system that can process and categorize billions of images efficiently.

146. **[Tesla] Autonomous Vehicle Perception System**  
  Develop a system for autonomous vehicle perception that can process sensor data in real-time and make safe driving decisions.

147. **Personalized News Feed Ranking**  
  Develop a personalized news feed ranking system that can handle millions of users and articles while maintaining freshness and relevance.

148. **Large Language Model Training System**  
  You're designing a distributed training system for large language models. How would you implement model parallelism and data parallelism? What are the trade-offs between them?

149. **Feature Store Design**  
  For a recommendation system that needs to serve millions of users, how would you design the feature store? Consider aspects like real-time vs. batch features, storage choices, and serving architecture.

150. **Supply Chain Demand Forecasting**  
  Develop a system for demand forecasting in supply chain management, processing large volumes of historical sales data to predict future demand accurately.

151. **Reinforcement Learning Training System**  
  Design a scalable reinforcement learning system for training autonomous agents in complex simulated environments.

152. **Genomic Analysis Platform**  
  Design a large-scale genomic analysis platform that uses machine learning to identify disease markers and predict patient outcomes from DNA sequencing data.

153. **Real-time Speech Recognition**  
  Design a real-time speech recognition system that can transcribe live audio streams with high accuracy and support multiple languages.

154. **MLOps Infrastructure**  
  Can you describe the key steps involved in creating infrastructure for MLOps?

155. **[Startup] Concept Drift Detection and Fixes**  
  When could concept drift occur? How do you detect and address model or concept drift in deployed machine learning models?

156. **MLOps Data Privacy**  
  What strategies can be implemented to ensure data privacy and security in MLOps pipelines?

157. **MLOps Scalability**  
  What strategies do you employ to ensure scalability in your MLOps processes?

158. **Infrastructure as Code in MLOps**  
  Can you explain the concept of Infrastructure-as-Code (IaC) and its role in MLOps?

159. **Model Bias Detection in MLOps**  
  How do you detect and address model bias within an MLOps framework?

160. **Batch vs Real-time Inference**  
  What distinguishes batch inference from real-time inference in machine learning applications?

161. **Search Engine Related Queries**  
  When a user enters a search query on a search engine like Google, a list of related searches is displayed to enhance the search experience. How would you design a system to generate relevant related search suggestions for each query?

162. **[Airbnb] Property Search**  
  How would you design a system to display the top 10 rental listings when a user searches for properties in a specific location on a platform like Airbnb?

163. **Trigger Word Detection**  
  How would you design an algorithm to accurately detect the trigger word 'activate' within a 10-second audio clip?

164. **Document QA System**  
  How would you design a question-answering system that can extract an answer from a large collection of documents given a user query?

165. **[Meta] Instagram Content Recommendations**  
  How would you design a machine learning pipeline for Instagram's content recommendation system to handle data freshness, mitigate novelty effects in A/B testing, and ensure personalized content delivery to over a billion users?

166. **Food Delivery Query Understanding**  
  How would you design a machine learning system to understand and expand user queries for a food delivery platform like Uber Eats, addressing challenges such as building a domain-specific knowledge graph, implementing representation learning for query expansion, handling ambiguous user intent, and ensuring real-time performance?

167. **Soft Prompt Tuning for Large Language Models**  
  Explain how soft prompt tuning can be utilized to adapt a large language model to a new NLP task without modifying the model's core parameters. Discuss the benefits and potential challenges associated with this approach.

168. **Caching Strategies for Large Language Models**  
  How would you design a caching system for serving large language model (LLM) responses to reduce latency and cost, while ensuring the accuracy and reliability of the responses? Discuss the key components of your caching mechanism, how you would handle semantic similarity, and the potential challenges you might face.

169. **Implementing Defensive UX for LLM-Based Products**  
  How would you design a Defensive UX strategy for a product that utilizes large language models (LLMs) to anticipate and gracefully handle errors, guide user behavior, and prevent misuse, while ensuring accessibility, trust, and a seamless user experience?

170. **Implementing Cascade Pattern in ML Systems**  
  How would you design a machine learning system using the cascade pattern to solve a complex problem by breaking it down into smaller, sequential tasks? Discuss the advantages and potential drawbacks of this approach, and provide real-world examples of its application.

171. **[Openai] Human-In-The-Loop for Collecting Explicit Labels**  
  How would you design a Human-In-The-Loop (HITL) system to collect explicit labels for a supervised learning task, balancing the need for high-quality annotations with the constraints of cost and scalability? Discuss the methods you would use, the advantages and disadvantages of HITL, and how you might leverage large language models to enhance the labeling process.

172. **[Openai] Reframing Machine Learning Problems for Enhanced Performance**  
  How would you apply the reframing technique to simplify a machine learning problem or its labels to improve model performance? Provide two examples of successful reframing and discuss the potential challenges of this approach.

173. **Evaluating LLM Outputs with LLM-Evaluators**  
  How would you design and implement a system that uses large language models (LLMs) as evaluators to assess the quality, correctness, and safety of another LLM's responses, considering factors such as evaluation methods, alignment with human judgments, and scalability?

174. **Aligning LLM-Evaluators to User-Defined Criteria**  
  How would you design a system to align large language model (LLM) evaluators with user-defined criteria, ensuring accurate and reliable assessments of another LLM's responses? Discuss the methods for defining and refining evaluation criteria, the role of interactive systems in this alignment, and the challenges involved in maintaining consistency with human judgments.

175. **Advancements in Evaluation Metrics for Machine Learning Models**  
  How do newer evaluation metrics such as BERTScore and MoverScore address the challenges posed by traditional metrics in evaluating machine learning models?

176. **Mitigating Biases in Automated Evaluations Using Large Language Models**  
  What strategies can be employed to mitigate biases in automated evaluations using large language models, and how do these strategies enhance the reliability of model assessments?

177. **Enhancing Open-Domain QA Systems with Fusion-in-Decoder (FiD)**  
  How does Fusion-in-Decoder (FiD) enhance open-domain question-answering systems, and what are its key advantages over other retrieval-based models?

178. **Internet-Augmented LLM System Design**  
  How do internet-augmented language models utilize search engines to enhance their performance in question-answering tasks, and what are the key components and processes involved?

179. **RAG System Design with Hybrid Retrieval**  
  How can machine learning teams effectively apply Retrieval-Augmented Generation (RAG) using hybrid retrieval methods, and what are the key considerations and technologies involved in optimizing retrieval for large-scale applications?

180. **[Royal Bank Canada] Test if it's Gaussian**  
  How would you determine that your dataset follows a Gaussian distribution? What if the data is univariate vs multivariate?

181. **[Microsoft] T-distribution vs normal distribution**  
  When is t-distribution used as opposed to normal distribution? How many data points are considered good enough to use for a z vs a t test?

182. **[Microsoft] Linear regression assumptions test**  
  The assumptions of linear regression are known to be: linearity, independence of errors, homoscedasticity, normality of residuals and no multicollinearity. How is each assumption tested?

183. **[Google Deepmind] Strong scaling**  
  Can you explain, in 1 sentence, the concept of strong scaling in the context of large language models?

184. **[Google Deepmind] Possible Scenario**  
  When a GPU or TPU rated at 500 TFLOPS only sustains ~50 TFLOPS on a large-model kernel, what possible scenarios related to chip-level factors can explain this tenfold performance drop?

185. **[Google Deepmind] Total memory computation.**  
  Let A be an array with shape int8[128, 2048] sharded as A[I_XY, J] over a devide mesh Mesh({‘X': 2, ‘Y': 8, ‘Z': 2}) (so 32 devices total). How much memory does A use per device? How much total memory does A use across all devices?

186. **[OpenAI] Total number of parameters.**  
  Let a transformer with D=4096 (hidden size) and F=4D (width of the feed-forward layers), V=32000 (size of model's vocabulary) and L=64 (depth of the network). You can assume we have multi-head attention with int8 KVs where the hidden size D is split accross N heads, each of the size H. How many parameters does the model have and what fraction of these are attention parameters?

187. **[OpenAI] Tricks to improve generation throughput.**  
  Mention 4 methods/tricks to improve generation throughput and latency in the context of large language models.

188. **[OpenAI] Mixing in Local Attention Layers**  
  How would you interleave local-window attention with global attention to curb KV-cache growth at long contexts? What are the pros and cons?

189. **[Meta] Data and model parallelism**  
  How does reducing the per-device batch size—especially when scaling out with fully-sharded data parallelism (FSDP)—impact the communication cost of 1D tensor (model) parallelism compared to data parallelism?
