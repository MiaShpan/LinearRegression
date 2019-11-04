package HomeWork1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class MainHW1 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		// load data
		Instances trainingData = loadData("src/HomeWork1/wind_training.txt");
		// find best alpha and build classifier with all attributes
		Instances testingData = loadData("src/HomeWork1/wind_testing.txt");
		LinearRegression windModel = new LinearRegression();
		windModel.buildClassifier(trainingData);
		// getting the calculated alpha
		double alpha = windModel.getAlpha();
		// calculate the error with  the training data with all attributes
		double training_error = windModel.calculateMSE(trainingData);
		// calculate the error with  the testing data with all attributes
		double testing_error = windModel.calculateMSE(testingData);

		System.out.println("The chosen alpha is: " + alpha);
		System.out.println("Training error with all features is: " + training_error);
		System.out.println("Test error with all features is: " + testing_error);



		System.out.println("List of all combinations of 3 features and the training error:");
		// build classifiers with all 3 attributes combinations
		LinearRegression windModelTrio = new LinearRegression();
        int[] minTrioIndexes = new int[3];
		int[] trioIndexes = new int[4];
		// keeps y in every instance
		trioIndexes[3] = windModel.getClassIndex();
		Remove filterToTrio = new Remove();
		double min_trio_error = Double.POSITIVE_INFINITY;
		double current_trio_error;
		double testing_trio_error;

		// Iterating over all 3 features combinations
		for(int i = 0; i < windModel.getClassIndex(); i++)
		{
		    trioIndexes[0] = i;
			for(int j=i+1; j < windModel.getClassIndex(); j++)
			{
                trioIndexes[1] = j;
				for(int k=j+1; k<windModel.getClassIndex(); k++)
				{
					trioIndexes[2] = k;

                    // filtering by the current trio
					filterToTrio.setAttributeIndicesArray(trioIndexes);
                    // in order to keep the attributes in the following indexes
                    filterToTrio.setInvertSelection(true);
                    filterToTrio.setInputFormat(trainingData);
					Instances trio = Filter.useFilter(trainingData,filterToTrio);
					// building a model with the current trio
					windModelTrio.buildClassifier(trio,windModel.getAlpha());
					// calculating current trio error
					current_trio_error = windModelTrio.calculateMSE(trio);

					System.out.println("Features: " + trainingData.attribute(i).name() 
						+ ", " + trainingData.attribute(j).name() + ", " + trainingData.attribute(k).name() 
						+ " training error: " + current_trio_error);

					// the new error is smaller than the minimum error until now
					if(current_trio_error < min_trio_error)
					{
						// saves attributes indexes
						for (int l=0; l<minTrioIndexes.length; l++)
						{
							minTrioIndexes[l] = trioIndexes[l];
						}
						// saves the minimum error until now
						min_trio_error = current_trio_error;
					}
				}
			}
		}

		System.out.println("Training error the features " + trainingData.attribute(minTrioIndexes[0]).name() 
			+ ", " + trainingData.attribute(minTrioIndexes[1]).name() + ", " 
			+ trainingData.attribute(minTrioIndexes[2]).name() + ": " + min_trio_error);

		// saving the best attributes in a new array
        for (int l=0; l<minTrioIndexes.length; l++)
        {
            trioIndexes[l] = minTrioIndexes[l];
        }
        // keeping only the best attributes
        filterToTrio.setAttributeIndicesArray(trioIndexes);
        // in order to keep the attributes in the following indexes
        filterToTrio.setInvertSelection(true);
        filterToTrio.setInputFormat(testingData);
        // for checking the error on the testing data with the best 3 attributes
        Instances trio_testing = Filter.useFilter(testingData,filterToTrio);
        // for training our algorithm with the training attributes
        Instances trio_training = Filter.useFilter(trainingData,filterToTrio);
        // training our algorithm with the training data
        windModelTrio.buildClassifier(trio_training,windModel.getAlpha());
        // calculating current trio error on the testing data
        testing_trio_error = windModelTrio.calculateMSE(trio_testing);
        System.out.println("Test error the features " + testingData.attribute(minTrioIndexes[0]).name() + ", " + testingData.attribute(minTrioIndexes[1]).name() + ", " + testingData.attribute(minTrioIndexes[2]).name() + ": " + testing_trio_error);
	}
}
