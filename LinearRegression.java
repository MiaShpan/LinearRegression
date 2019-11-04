package HomeWork1;//package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;

public class LinearRegression implements Classifier {
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;
	
	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights. 
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes()-1;
		//finding alpha only when building a model using all the data
		findAlpha(trainingData);
		m_coefficients = gradientDescent(trainingData);
	}

	//the method which runs to train the linear regression predictor, i.e
	//finds its weights, when alpha was already found
	public void buildClassifier(Instances trainingData, double alpha) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes()-1;
		m_alpha = alpha;
		m_coefficients = gradientDescent(trainingData);
	}
	
	private void findAlpha(Instances data) throws Exception {
		double min_error = Double.POSITIVE_INFINITY;
		double min_alpha = 0, prev_error, current_error;
		// going throw possible alpha values
		for (int i = -17; i < 1; i++){
			guessTheta();
			m_alpha = Math.pow(3, i);
			prev_error = Double.POSITIVE_INFINITY;
			// run gradient descent 20000 (200*100) times
			for (int j = 0; j < 200; j++) {
				// every 100 iterations calculate the error
				for (int k = 0; k < 100; k++) {
					updateTheta(data);
				}
				current_error = calculateMSE(data);
				// the previous error is smaller than the current
				if (prev_error < current_error) {
					break;
				} else {
					prev_error = current_error;
				}
			}
				// after 20000 iterations (if no breaks) the minimal error is the last calculated error
				// the minimal error until now is bigger than the new error
				if(min_error > prev_error)
				{
					// update minimal error to new error
					min_error = prev_error;
					// update the alpha of the minimal error to current alpha
					min_alpha = m_alpha;
				}
			}
		// the alpha that gave us the minimal error is min_alpha
		m_alpha = min_alpha;
	}

	// gussing some theta value 
	private void guessTheta()
	{
		m_coefficients = new double[m_truNumAttributes+1];
		for(int i = 0; i < m_coefficients.length; i++)
		{
			m_coefficients[i]=1;
		}
	}
	
	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */

	private double[] gradientDescent(Instances trainingData)
			throws Exception {
		double lastError = Double.POSITIVE_INFINITY;
		double error;
		// guessing some theta
		guessTheta();

		//keep going until the error difference is small enough 
		while (true) {
			// every 100 iteration calculate the error
			for (int j=0; j<100; j++) {
				updateTheta(trainingData);
			}
			error = calculateMSE(trainingData);
			// the difference between the errors is smaller than 0.003
			if(Math.abs(lastError - error) < 0.003)
			{
				return m_coefficients;
			} else {
				lastError = error;
			}
		}
	}

	private void updateTheta(Instances data) throws Exception{
		//creating a temp array to save the updated theta values
		double[] updatedValues = new double[data.numAttributes()];
		// for every theta
		for (int i = 0; i < m_coefficients.length; i++){
			// theta = theta - (alpha * derivative)
			updatedValues[i] = m_coefficients[i] - (m_alpha * derivative(i, data));
		}
		//updating m_coefficients
		for (int i = 0; i < m_coefficients.length; i++){
			// theta = theta - (alpha * derivative)
			m_coefficients[i] = updatedValues[i];
		}
	}

	// calculate the derivative by theta with index -> index
	public double derivative(int index, Instances data) throws Exception{

		double sum = 0;
		// calculates the derivative by theta 0
		if (index == 0){
			// for every instance 
			for (int i = 0; i < data.numInstances(); i++){
				// sum = sum + (h(instance(i)) - y(i))
				sum += (regressionPrediction(data.instance(i))) -  (data.instance(i).value(m_ClassIndex));
			}
			// sum : m 
			return sum/data.numInstances();
			// calculate the derivative by theta > 0
		} else {
			// for every instance 
			for (int j = 0; j < data.numInstances(); j++){
				// sum = sum + (h(instance(i)) - y(i)) *  artibute(index - 1)
				sum += (regressionPrediction(data.instance(j)) -  data.instance(j).value(m_ClassIndex)) * data.instance(j).value(index-1);
			}
			// sum : m 
			return sum/data.numInstances();
		}
	}

	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		// adding theta 0
		double prediction = m_coefficients[0];
		for (int i = 0; i < m_truNumAttributes; i++){
			// sum + theta(i) * x(i)
			prediction += instance.value(i) * m_coefficients[i + 1];
		}
		return prediction;
	}
	
	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param data
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {
		double sum=0;
		// for every instance
		for(int i=0; i<data.numInstances(); i++)
		{
			// sum = sum + (h(xi) - yi) ^2
			sum += Math.pow(regressionPrediction(data.instance(i)) - data.instance(i).value(m_ClassIndex),2);
		}
		// sum : 2m
		return sum/(data.numInstances()*2);
	}

	public double getAlpha()
	{
		return this.m_alpha;
	}

	public int getClassIndex()
	{
		return this.m_ClassIndex;
	}
    
    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
}
