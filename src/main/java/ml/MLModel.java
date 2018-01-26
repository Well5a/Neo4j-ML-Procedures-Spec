package ml;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Abstract class that defines the models for Machine Learning.
 * 
 * @author mwe
 */
public abstract class MLModel 
{
	/** The Map that stores the created MLModels. */
	static ConcurrentHashMap<String,MLModel> models = new ConcurrentHashMap<>();
	
	/** The name of the model. */
	final String name;	
	
	/** The names of features and their data types. */
	final Map<String, Types> types = new HashMap<>();
	
	/** The number of features of the model. */
	final int nFeatures;
	
	/** The rows that are added to the model and trained upon. */
	List<List<String>> rows;
	
	/** The number of added rows. */
	int nRows;
	
	/** The state of the model. */
	State state;
	
	/** The Map with extra attributes. */
	Map<String, String> extra;
	
	/**
	 * Instantiates a new MLModel.
	 * Stores a new model in the ConcurrentHashMap <tt>models</tt>.
	 * Sets the state of the model to "created".
	 *
	 * @param name the name of the model
	 * @param types the Map of attribute names and their respective data types
	 * @param extra the Map with extra attributes
	 */
	public MLModel(String name, Map<String, String> types, Map<String, String> extra) 
	{
		if (models.containsKey(name)) throw new IllegalArgumentException("Model "+name+" already exists, please remove first");
		
        this.name = name;
        for (Map.Entry<String, String> entry : types.entrySet()) 
        {
            this.types.put(entry.getKey(), Types.getTypes(entry.getValue()));
        }
        
        this.nFeatures  = types.size();
        this.rows 		= new ArrayList<List<String>>();
		this.nRows 		= 0;
		this.state		= State.created;
		this.extra 		= extra;
		
        models.put(name, this);
	}
	
	/**
	 * Creates a new MLModel.
	 * Calls the constructor of subclass <tt>Nd4jImplementation</tt>.
	 *
	 * @param name the name of the model
	 * @param types the Map of attribute names and their respective data types
	 * @param params the hyperparameters for the specific Machine Learning implementation
	 * @param extra the Map with extra attributes
	 * @param timePeriod the boolean that indicates whether a time series shall be predicted or not
	 * @return the MLModel that has been created
	 */
	public static MLModel create(String name, Map<String, String> types, Map<String, Object> params, Map<String, String> extra, boolean timePeriod, String implementation)
	{
		switch(Implementations.getImplementations(implementation))
		{
			case Nd4j:
				return new Nd4jLinReg(name, types, params, extra, timePeriod);
			case Dl4j:
				return new Dl4jRNN(name, types, params, extra, timePeriod);
			default:
				throw new IllegalArgumentException("Unknown Implementation: " + implementation);
		}
	}
	
	/**
	 * Constructs a row from the given features and the prediction value and adds it to the rows of the model.
	 * Sets the state of the model to "filled".
	 *
	 * @param features the features
	 * @param value the prediction value (label)
	 */
	public void add(Map<String, Object> features, Object value)
	{
		List<String> row = new ArrayList<>();
		
		//Add the label to the first column of the row
		row.add(value.toString());
		
		//Add the features 
		for(String key : types.keySet())
		{
			if(features.get(key) == null) throw new IllegalArgumentException("The featurename '"+key+"' specified for this model is not given in this add call. Call ml.info.");
						
			String val = features.get(key).toString();
			if(types.get(key) == Types._Numeric)
			{
				row.add(val);
			}
			else //for Types.Class
			{
				if(ClassAttribute.classAttributes.get(key) == null) //if attribute has not been added (only for the first row that gets added to the model)
				{
					new ClassAttribute(key);
				}
				ClassAttribute classAttribute = ClassAttribute.classAttributes.get(key); //get the instance of the attribute
				row.add(classAttribute.getValue(val));
			}
		}
		rows.add(row);
		nRows++;
		this.state = State.filled;
	}
	
	/**
	 * Trains the model on the added data.
	 * Sets the state of the model to "trained".
	 *
	 * @throws IOException Signals that an I/O exception has occurred.
	 * @throws InterruptedException the interrupted exception
	 */
	protected abstract void train() throws IOException, InterruptedException;
	
	/**
	 * Predicts values from the given features on a trained model.
	 *
	 * @param features the features
	 * @return the Collection of different features and their predicted values
	 * @throws InterruptedException 
	 * @throws IOException 
	 */
	protected abstract Collection<Map<String, Object>> predict(Map<String, List<Object>> features) throws IOException, InterruptedException;

	/**
	 * Removes the model from <tt>models</tt>.
	 *
	 * @param name the name of the model
	 * @return the RowResult 
	 */
	public static ML.RowResult remove(String name) 
	{
        MLModel existing = models.remove(name);
        if (existing != null) return new ML.RowResult("Removed Model: '"+name+"'");
        throw new IllegalArgumentException("No valid ML-Model " + name);
    }
	
	/**
	 * Gets the instance of the model.
	 *
	 * @param name the name of the model
	 * @return the MLModel
	 */
	public static MLModel getModel(String name) 
	{
        MLModel model = models.get(name);
        if (model != null) return model;
        throw new IllegalArgumentException("No valid ML-Model " + name);
    }

	/**
	 * Gets the data that has been added to the model.
	 * Every String in this List represents a row.
	 *
	 * @return the rows
	 */
	public List<String> getRows()
	{
		List<String> result = new ArrayList<>();
		Iterator<List<String>> rowIter = rows.iterator();
		while(rowIter.hasNext())
		{
			result.add(rowIter.next().toString());
		}
		
		return result;
	}
	
	/**
	 * Gets general information about the model and the specific implementation.
	 *
	 * @return the List of Information about specific Objects
	 */
	protected abstract List<Object> getSpecials();
	
	
	/**
	 * Returns a List of names of all Models that have been created.
	 * 
	 * @return the List of Model names
	 */
	public static List<String> getAllModels()
	{
		List<String> modelNames = Collections.list(models.keys());
		Collections.sort(modelNames);
		if(modelNames.isEmpty())
		{
			modelNames.add("There are currently no models in the database");
		}
		return modelNames;
	}
	
	/**
	 * The Enum State holds the possible values for the class variable <tt>state</tt> 
	 * which is set to different States to confirm what has already been done with the model.
	 * If new data is added to a trained model it has to be trained again.
	 */
	public enum State 
	{	
		/** The created State indicates that the model has been initialized but has no training data. */
		created, 
		/** The filled State indicates that data has been added to the model and it is ready to train. */
		filled, 
		/** The trained state indicates that the model has been trained and is ready to predict. */
		trained
	}
	
	/**
	 * The Enum Types defines the possible data types for the features.
	 */
	public enum Types 
	{			
		/** The Numeric data type(Integer, Float, Double, etc.). */
		_Numeric, 
		/** The Class data type (Boolean, String, etc.). */
		_Class;
		
		/**
		 * Gets the types.
		 *
		 * @param type the type
		 * @return the types
		 */
		public static Types getTypes(String type) 
		{
            switch (type.toUpperCase()) 
            {
                case "NUMERIC":
                    return Types._Numeric;
                case "CLASS":
                    return Types._Class;
                default:
                    throw new IllegalArgumentException("Unknown type: " + type);
            }
		}
	}
	
	public enum Implementations
	{
		Nd4j, 
		Dl4j;
		public static Implementations getImplementations(String implementation) 
		{
            switch (implementation.toUpperCase()) 
            {
                case "ND4J":
                    return Implementations.Nd4j;
                case "DL4J":
                    return Implementations.Dl4j;
                default:
                    throw new IllegalArgumentException("Unknown Implementation: " + implementation);
            }
		}
	}
	
	/**
	 * Stores the possible values for attributes of type <tt>Class</tt> and a numeric representation.
	 * The latter can be used for training and prediction.
	 */
	protected static class ClassAttribute
	{
		/** The Map that stores the different attributes of type class. */
		static ConcurrentHashMap<String,ClassAttribute> classAttributes = new ConcurrentHashMap<>();
		
		/** The Map with the original values of the attribute and their numeric representations. */
		Map<String, Integer> valuesMap;
		
		/** The numeric representation for the next new attribute value. */
		Integer newNumValue;
		
		/** The name of the attribute. */
		String name;
		
		/**
		 * Instantiates a new class attribute.
		 *
		 * @param name the name of the attribute
		 */
		public ClassAttribute(String name)
		{
			this.name = name;
			this.newNumValue = 0;
			valuesMap = new HashMap<>();
			classAttributes.put(name, this);
		}
		
		/**
		 * Gets the numeric value for a class attribute as String.
		 *
		 * @param val the name of the attribute
		 * @return the numeric value
		 */
		protected String getValue(String val)
		{
			//if this is the first time this value gets passed, store it and its numeric representation in the "valuesMap"
			if(valuesMap.get(val) == null)
			{
				valuesMap.put(val, newNumValue);
				newNumValue++; //set every value to zero so it will be ignored for prediction
			}
			return valuesMap.get(val).toString(); //return the numeric representation (as String)
		}
	}
}


	