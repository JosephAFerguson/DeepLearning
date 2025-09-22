[HW1](/HW1)
https://github.com/JosephAFerguson/DeepLearning/edit/main/README.md
<details>
  My homework report is named as HW1_Report_fergujp.pdf and is in this folder.
  My code is called HW1.py and is in this folder.

  TO RUN THE CODE:
    You need torch, matplotlib, scikit-learn, pandas, and numpy(although this is installed as a dependency for other libaries) installed and python3 installed.
    At the bottom you can add your tests under the if __name__=="__main": statement, or if you import just use test_model(params)

    The params are as follows:

      model_architexture = 
        Choose between "Linear Regression", "DNN-16", "DNN-30-8", "DNN-30-16-8", "DNN-30-16-8-4" and "DNN-8-4".

      activationFunction = 
        Choose between ReLU, LeakyReLU, Sigmoid, or Tanh. Do not use strings these are variables in the code

      Batchsize = 
        Choose a batch size as an int

      Learning rate =
        Choose a learning rate as an int

      Epochs = 
        Choose the number of epochs as an int

  For highest-performed DNN model and Linear regression model, there will be screenshots in /ScreenShots
  For screenshot's of iteration of model's training and testing with timestamps, it will be in /Screenshots as well

  Note: I was not able to replicate the one 0.87 R2 score I got for the 30-16-8-4 high performed model after adding timestamps to the code, this also explains the small inconsistencies between the report and the screenshots
</details>
