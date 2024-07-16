# Cinema-Sentiment-Analyzer

This project is a web-based application designed to analyze and predict the sentiment of movie reviews using machine learning techniques. Developed with Flask for backend functionality and HTML, CSS, and JavaScript for the frontend interface, the application allows users to input movie reviews and receive accurate sentiment predictions.

## Project Description

The Cinema Sentiment Analyzer leverages advanced machine learning techniques to accurately classify the sentiment of movie reviews using the IMDB dataset, which contains 50,000 reviews. This project aims to provide precise sentiment predictions, crucial for understanding audience reactions and aiding decision-making for both viewers and filmmakers. The dataset is divided into 25,000 reviews for training and 25,000 for testing, enabling robust model evaluation. The application utilizes a variety of algorithms, including Naive Bayes, Random Forest, Logistic Regression, and K-Nearest Neighbors, to predict the number of positive and negative reviews effectively. This tool is essential for natural language processing and text analytics in the entertainment industry.

## Features

- **User-friendly interface**: The frontend interface is designed to be intuitive and easy to navigate.
- **Accurate predictions**: The system utilizes machine learning algorithms to provide accurate predictions based on inputted movie reviews.
- **Responsive design**: The application is responsive and works seamlessly across different devices and screen sizes.

## Model Selection

The best model for predicting movie review sentiment is selected based on accuracy. The following models are evaluated:

- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem.
- **RandomForestClassifier**: Ensemble learning technique known for its robustness and accuracy.
- **Logistic Regression**: Classification algorithm used for binary outcomes.
- **KNeighborsClassifier**: Classification algorithm based on similarity to neighboring data points.

The accuracy of each model is calculated on the test dataset, and the Naive Bayes model is chosen as one of the models for prediction.

## Steps for Prediction

1. **Open the application in a web browser.**

   ![image](https://github.com/user-attachments/assets/1095f7b8-65b2-435e-b8ec-702efbea61d1)

2. **Fill out the form with the movie review text.**

   ![image](https://github.com/user-attachments/assets/798d5a21-f321-4240-99ba-3b4b79b7b2f8)

3. **Click the "Analyze" button to generate the sentiment prediction.**

4. **View the prediction result to see the sentiment of the movie review.**

   ![image](https://github.com/user-attachments/assets/65e8522a-c024-482b-b58c-4c60fdecd42f)


## Dependencies

- **Flask**
- **Python**: 3.9.5

## Installation

1. **Clone the repository:**

   ```sh
   git clone <repository-url>
   ```

2. **Run the Flask application**

   ```bash
    python app.py
    ```

3. **Open a browser and navigate to http://127.0.0.1:5000 to access the web interface**.

## Conclusion

The Cinema Sentiment Analyzer is a powerful tool designed to assist in understanding the sentiment of movie reviews. By leveraging machine learning techniques and a user-friendly interface, the application provides accurate predictions based on inputted text data.

With its responsive design and precise predictions, this system aims to empower individuals to make informed decisions about movies and improve their overall viewing experience. By identifying sentiment patterns and providing personalized insights, it contributes to better movie recommendations and enhances overall enjoyment.

The accuracy of the Cinema Sentiment Analyzer's prediction models ensures reliable and trustworthy results for users. This high level of accuracy instills confidence in the system's ability to make informed predictions and assist viewers in selecting movies that match their preferences.

As the project continues to evolve, future enhancements may include incorporating additional features, expanding the dataset for improved accuracy, and integrating with movie recommendation systems for a seamless user experience.

With a commitment to innovation and excellence, the Cinema Sentiment Analyzer remains dedicated to advancing movie sentiment analysis for the betterment of the entertainment industry.

## About the Developer

This project is developed by Abhishek Kushwaha.

- **LinkedIn**: https://www.linkedin.com/in/abhishek10027
- **GitHub**: https://github.com/abhishek10027
