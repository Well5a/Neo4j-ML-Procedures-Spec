package ml;

import java.io.IOException;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.ListStringSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;


/**
 * Implements Linear Regression on the class <tt>MLModel</tt> with the Nd4j framework.
 * 
 * @author mwe
 */
public class Nd4jLinReg extends MLModel 
{	
	/** The formatter of the date values. */
	private DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd");
		
	/** The boolean that indicates whether a time series shall be predicted or not. */
	final boolean timePeriod;
	
	/** The array with theta values for Linear Regression. */
	INDArray theta; 
	
	/** The array with mean values. */
	INDArray mean;
	
	/** The array with standard deviation values. */
	INDArray sigma;
	
	/**  The hyperparameters for Linear Regression. */
	Map<String, Object> params;
	
	/**  The cost of the trained model. */
	double cost;
		
	/**
	 * Instantiates a new MLModel.
	 *
	 * @param name the name of the model
	 * @param types the Map of attribute names and their respective data types
	 * @param params the hyperparameters for Linear Regression
	 * @param extra the Map with extra attributes
	 * @param timePeriod boolean that indicates whether a time series shall be predicted or not
	 */
	public Nd4jLinReg(String name, Map<String, String> types, Map<String, Object> params, Map<String, String> extra, boolean timePeriod) 
	{
		super(name, types, extra);
		
		//Set Nd4j to use double values
		DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
		
		this.mean		= Nd4j.zeros(1);
		this.sigma 		= Nd4j.zeros(1);
		this.cost 		= 0.0d;
		this.params		= params;
		this.timePeriod = timePeriod;
		
		//Initialize theta
		@SuppressWarnings("unchecked")
		List<Double> thetaStart = (List<Double>) params.get("theta");
		this.theta = Nd4j.zeros(thetaStart.size(), 1);
		for(int j = 0; j < thetaStart.size(); j++)
		{
			this.theta.put(j, 0, thetaStart.get(j));
		}
		
		//Check if the right number of theta values is given (number of feature columns plus one bias column)
		if(nFeatures+1 != theta.rows())
		{
			models.remove(name);
			throw new IllegalArgumentException("Illegal number of theta values. Expected " + (nFeatures+1) + " values got " + theta.rows() + ". Model '"+name+"' has not been created.");
		}
	}
		
	/* (non-Javadoc)
	 * @see ml.MLModel#train()
	 */
	@Override
	protected void train() throws IOException, InterruptedException
	{		
		if (this.state == State.created) throw new IllegalArgumentException("Model "+name+" has no training data, please add some before training.");
		
		//Initialize hyperparameters
		double alpha = (double) params.get("alpha");
		long iter 	 = (long) params.get("iter");
			
        //Load Data in arrays
		ListStringSplit input = new ListStringSplit(rows);
		ListStringRecordReader lsrr = new ListStringRecordReader();
        lsrr.initialize(input);
        //Parameters: data as ListStringRecordReader, number of rows, index of label column, number of possible label columns, boolean regression 
        DataSetIterator dataIter = new RecordReaderDataSetIterator(lsrr, nRows, 0, 0, true);
	      
        DataSet dataSet 	= dataIter.next();
        INDArray features 	= dataSet.getFeatureMatrix();
        INDArray labels 	= dataSet.getLabels();
          
		//Shape dates
		features = calcDateDiff(features, nRows, 0);
				
		//Feature Normalization
		mean	= features.mean(0);
		sigma 	= features.std(true, 0);	
		for(int j = 0; j < sigma.columns(); j++)
		{
			if(sigma.getDouble(0, j) == 0.0d) sigma.put(0, j, 1.0d);
		}
		features = normalizeFeatures(features, mean, sigma);
				
		//Add bias values
		features = addBiasValues(features, nRows);
								
		//Run Gradient Descent to compute optimal theta values
		theta 	= gradientDescent(features, labels, theta, alpha, iter, nRows);
		//theta = normalEquations(features, labels);
		cost	= Nd4j.sum(Transforms.pow((features.mmul(theta)).sub(labels), 2)).mul((1 / (2*nRows))).getDouble(0); //compute cost
		System.out.println("Theta found by Gradient Descent: " + theta);
		this.state = State.trained;
		return;
	}
	
	/* (non-Javadoc)
	 * @see ml.MLModel#predict(java.util.List)
	 */
	@Override
	protected Collection<Map<String, Object>> predict(Map<String, List<Object>> features)
	{
		if (this.state != State.trained) throw new IllegalArgumentException("Model "+name+" is not trained, please train first. If you have added some new data after the last training, train the model again before predicting.");
		
		//Number of values to predict (number of rows of "featuresArr")
		int nPredictions = 1;
		INDArray featuresArr;
		
		if(timePeriod) //if values for a time period shall be predicted
		{
			if(features.keySet().size() != 2) throw new IllegalArgumentException("Illegal number of feature values. Parameter for predicting a period is set to true, please provide exactly one start and end date as features.");
			LocalDate startDate = LocalDate.parse(String.valueOf(features.get("start")), formatter);
			LocalDate endDate 	= LocalDate.parse(String.valueOf(features.get("end")), formatter);
			
			List<String> dates = new ArrayList<>();
			for(LocalDate date = startDate; !date.isAfter(endDate); date = date.plusDays(1))
			{
				dates.add(date.toString().replaceAll("-", ""));
			}
			nPredictions = dates.size();
			featuresArr = Nd4j.zeros(nPredictions, 1);
			for(int i = 0; i < nPredictions; i++)
			{
				featuresArr.put(i, 0, Double.parseDouble(dates.get(i)));
			}
		}	
		else if(!(features.get(types.keySet().iterator().next()) instanceof List<?>)) //Checks whether the value of "features" is a single value and not a List
		{
			featuresArr = Nd4j.zeros(nPredictions, nFeatures);
			int colIter = 0;
			for (Entry<String, Types> entry : types.entrySet()) //iterate columns
			{
				if(entry.getValue() == Types._Class) //for attributes of data type class load the numeric value in the feature array
				{
					featuresArr.put(0, colIter, Double.parseDouble(String.valueOf(ClassAttribute.classAttributes.get(entry.getKey()).getValue(String.valueOf(features.get(entry.getKey()))))));
				}
				else
				{
					featuresArr.put(0, colIter, Double.parseDouble(String.valueOf(features.get(entry.getKey()))));
				}
				colIter++;
			}
		}
		else //for a List of rows, several predictions at once
		{
			nPredictions = features.get(types.keySet().iterator().next()).size();
			featuresArr = Nd4j.zeros(nPredictions, nFeatures); //columns := attributes, rows := prediction sets
			int colIter = 0;
			for (Entry<String, Types> entry : types.entrySet()) //iterate columns
			{
				if(entry.getValue() == Types._Class) //for attributes of data type class load the numeric value in the feature array
				{
					for(int j = 0; j < nPredictions; j++) //iterate through rows
					{
						featuresArr.put(j, colIter, Double.parseDouble(String.valueOf(ClassAttribute.classAttributes.get(entry.getKey()).getValue(String.valueOf(features.get(entry.getKey()).get(j))))));
					}
				}
				else
				{
					for(int j = 0; j < nPredictions; j++) //iterate through rows
					{
						featuresArr.put(j, colIter, Double.parseDouble(String.valueOf(features.get(entry.getKey()).get(j))));
					}
				}
				colIter++;
			}
		}	
		
		//Duplicate of featuresArr 
		INDArray foo = featuresArr.dup();
		
		System.out.println(featuresArr);
		
		//Shape features for prediction
		featuresArr = calcDateDiff(featuresArr, nPredictions, 0);
		featuresArr = normalizeFeatures(featuresArr, mean, sigma);
		featuresArr = addBiasValues(featuresArr, nPredictions);
		
		//Predict values
		INDArray prediction = featuresArr.mmul(theta);		
		
		//Load the features and their predicted results in a Map. The result collection is a list of this maps.		
		Collection<Map<String, Object>> result = new ArrayList<>();	
		Map<String, Object> rowResult = new HashMap<>();
		
		for(int j = 0; j < nPredictions; j++) //iterate through rows (prediction sets)
		{
			rowResult = new HashMap<>();
			Iterator<String> keyIter = types.keySet().iterator();
			for(int i = 0; i < foo.columns(); i++) //iterate through columns (different features)
			{
				rowResult.put(keyIter.next(), Math.round(foo.getDouble(j, i)));
			}
			rowResult.put("prediction", Math.round(prediction.getDouble(j, 0)));
			if(extra != null)
			{
				for (Map.Entry<String, String> entry : extra.entrySet())
				{
					rowResult.put(entry.getKey(), entry.getValue());
				}
			}
			result.add(rowResult);
		}
		return result;
	}
	
	/**
	 * Gets an array of dates and returns an array of their difference in days to the current day.
	 *
	 * @param arr the INDArray of dates
	 * @param length the number of dates
	 * @param pos the index of the column with date values
	 * @return the INDArray of date differences
	 */
	INDArray calcDateDiff(INDArray arr, double length, int pos)
	{
		for(int i = 0; i < length; i++)
		{		
			arr.put(i, pos, ChronoUnit.DAYS.between(LocalDate.parse(String.valueOf(arr.getInt(i, pos)), formatter), LocalDate.now()));
		}
		return arr;
	}
	
	/**
	 * Normalizes features.
	 *
	 * @param arr the INDArray of features
	 * @param mean the mean
	 * @param sigma the standard deviation
	 * @return the INDArray with normalized values
	 */
	INDArray normalizeFeatures(INDArray arr, INDArray mean, INDArray sigma)
	{
		return (arr.subRowVector(mean)).divRowVector(sigma);
	}
	
	/**
	 * Adds a column of ones as the first column to the feature array. 
	 *
	 * @param arr the INDArray of features
	 * @param length the number of features
	 * @return the INDArray with features and additional bias column
	 */
	INDArray addBiasValues(INDArray arr, int length)
	{
		return Nd4j.hstack(Nd4j.ones(length, 1), arr);
	}
	
	/**
	 * Prints the cost for these theta values.
	 * Loss function 'mean squared error'.
	 *
	 * @param features the INDArray of features
	 * @param labels the INDArray of respective true values
	 * @param theta the parameters for the linear regression line
	 * @param length the number of features/values
	 */
	void computeCost(INDArray features, INDArray labels, INDArray theta, double length)
	{
		System.out.println("Cost for theta" + theta + ": " + Nd4j.sum(Transforms.pow((features.mmul(theta)).sub(labels), 2)).mul((1 / (2*length))));
	}
	
	/**
	 * Gradient Descent algorithm that computes optimal theta values.
	 *
	 * @param features the INDArray of features
	 * @param labels the INDArray of respective true values
	 * @param theta the parameters for the linear regression line
	 * @param alpha the step size
	 * @param iter the number of iterations
	 * @param length the number of features/values
	 * @return the INDArray of computed theta values
	 */
	INDArray gradientDescent(INDArray features, INDArray labels, INDArray theta, double alpha, long iter, double length)
	{
		INDArray foo;
		System.out.println("Running Gradient Descent:");
		for(int i = 1; i <= iter; i++)
		{
			foo = features.mulColumnVector((features.mmul(theta)).sub(labels));
			theta = theta.sub(((Nd4j.sum(foo, 0)).mul(1/length).transpose()).mul(alpha));
			computeCost(features, labels, theta, length);
		}
		return theta;
	}
	
	/**
	 * Normal Equations algorithm: alternative to compute optimal theta values.
	 *
	 * @param features the INDArray of features
	 * @param labels the INDArray of respective true values
	 * @return the INDArray of computed theta values
	 */
	INDArray normalEquations(INDArray features, INDArray labels)
	{
		return ((InvertMatrix.invert(features.transpose().mmul(features), false)).mmul(features.transpose())).mmul(labels);
	}
	
	
	/* (non-Javadoc)
	 * @see ml.MLModel#getParamsAsString()
	 */
	@Override
	protected List<Object> getSpecials() 
	{
		String extraString = "none";
		if(extra != null) extraString = extra.toString();
		
		return Arrays.asList(
		  "Status: " 				+ state
		, "Time series given: " 	+ timePeriod
		, "Feature names: " 		+ types.toString() + " (total: "+nFeatures+")"
		, "Extra attributes: " 		+ extraString
		, "Number of added rows: " 	+ nRows
		, "Theta: " 				+ theta.toString()
		, "Cost: " 					+ String.valueOf(cost)
		, "Mean: " 					+ mean.toString()
		, "Sigma: " 				+ sigma.toString()
		);
	}
}