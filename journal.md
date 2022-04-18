03/10/2022

- Today I researched approximate inference techniques. I have found at least 5 inference techniques I wish to consider assessing for my project. Gibbs sampling, MAP, 2 Markov Chain methods i.e Hamiltonian Monte Carlo and its predeccessor Metropolis Hastings, but these are not novel techniques. Im still looking for novel techniques to incorporate in my research. Might test a total of 7 approximation techniques.

- Ive decided to incorporate creation of my own baseyian model into my project. 

- May choose 3 different datasets for and create 3 different models as I also want to incorporate how a change in data type might affect the inference technique. I should also incorporate a pre made model.

- Also spent the day gathering references.

- * Set task for tomorrow :
    Begin Beysian model construction and write up intro
    
    
DISCONTINUED---------------------------------------------------------------------------------------------------------------------------------------------->
    
 03/15/2022
 
 - Ive decided to switch project ideas. Ive been running into some issues running code and have even had to purchase COlab pro to no success. 
 - I think going forward I will be developing models although im not quite sure what I want to make.
 - I want to develop something along the lines of a bayesian neural network.

 03/19/2022
 
 - Ive decided to create a generative models for 3 types of datasets, MNIST, Cats and Dogs, Pokemon, three fundementally different types of data. The reason I chose these datasets is because the MNIST holds relativley simple structures whose features are well documented and easy to learn, the second datset because their structures are a little harder to learn as cats and dogs have alot more variation in the data and lastly I wanted to see how well the model would preform on completeley unstructured data. Cartoons like pokemon have almost no structure as they can be anything or in any shape. It would be very useful to the animation industry to be automatically able to generate recognizably "structured cartoons" that means cartooons we can recognize as not being "nonesense".
 - I plan to develop a standard DCGAN
 - I plan to spend the week sanitizing data for my new project. Largely because I have alot of other work too but also because the dataset is rather huge.
 
 03/23/2022
 
 - I decided to add a bayesian treatment to the DCGAN
 - Ran into CUDA issues trying to run the GAN on this dataset. 
 - The standard pokemon image sizes are 256 x 256. I may need to shrink these. Decided to start with the MNIST dataset.

03/25/2022

- Completed prototype for standard DCGAN. Working on some implementation issues and making my code neater.
- Realized there are issues adding data to github so ive decided to just update my journal for now. 
- Currently confused about how im going to implement the bayesian treatment but ive been asking around and gotten a little bit of help so it shouldnt be too bad.

03/26/2022

- Finished developing the initial code for the standard GAN. Plan to finish optimizing the code and getting required computing resources to run it on a larger dataset or I will have to resize the poke_set images in order to fit it to the model. However due to the fact most of the code for the first GAN has been completed I intend to use the rest of the week to not only optimize but do research on how to begin implementing the next phase.

03/27/2022

- Lot of deadlines this week so I didnt do too much today. But I cant seem to figure out the issue with github. I cant seem to add or commit from colab. I will probably just have to include this in the milestone and final report. ALternative methods to consider are updating the journal manually and simply downloading updates and uploading from the terminal. Will have to see how this goes.

04/02/2022

- Been reading on Bayesian treatments and Bayesian neural networks. Its been pretty interesting to see results from BAyesian neural networks. Feeling positive on what it can do in the space of DCGAN's and generative models.

