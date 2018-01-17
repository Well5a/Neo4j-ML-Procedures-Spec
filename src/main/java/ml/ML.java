package ml;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.logging.Log;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

/**
 * Provides stored procedures for machine learning on the Neo4j database.
 * See the attached README for more Information on how to use them.
 *
 * @author mwe
 */
public class ML 
{
	/** The database service. */
	@Context
    public GraphDatabaseService db;

    /** The log. */
    @Context
    public Log log;
       
    /**
     * Creates a new model. 
     * Example: <code>CALL ml.create("modelName", {attributeName: "numeric"}, {alpha: 0.1}, {extraAttributeName: "true"}, true)</code>
     *
     * @param model the name of the model
     * @param types the Map of attribute names and their respective data types
     * @param params the hyperparameters for the specific Machine Learning implementation
     * @param extra the Map of attributes with constant values the model shall not be trained with but displayed with the result of the prediction
     * @param timePeriod the boolean that indicates whether a time series shall be predicted or not
     * @return the stream of RowResult
     */
    @Procedure
    public Stream<RowResult> create(@Name("model") String model
    								, @Name("types") Map<String, String> types
    								, @Name("params") Map<String, Object> params
    								, @Name("extra") Map<String, String> extra
    								, @Name("period") boolean timePeriod
    								, @Name("implementation") String implementation)
    {
    	MLModel.create(model, types, params, extra, timePeriod, implementation);
        return Stream.of(new RowResult("Created Model: '" + model + "'"));
    }
    
    /**
     * Adds training data to the model. A match before the call of this procedure is required to select the data.
     * <p>
     * The keys of the Map <tt>features</tt> must be the same as the ones of Map <tt>types</tt> from the "create" call.
     * <p>
     * Example for one feature called "date":
     * <code>
     * MATCH (n:User) CALL ml.add('user', {date: n.date}, n.count) YIELD result RETURN result
     * </code>
     * <p>
     * The procedure is called once for every matched row.
     *
     * @param model the name of the model
     * @param features the Map of featureNames and their respective names
     * @param label the label 
     * @return the stream of RowResult
     */
    @Procedure
    public Stream<RowResult> add(@Name("model") String model, @Name("features") Map<String, Object> features, @Name("label") Object label) 
    {
    	MLModel mlModel = MLModel.getModel(model);
    	mlModel.add(features, label);
        return Stream.of(new RowResult(mlModel.getRows().get(mlModel.nRows-1).toString()));
    }
    
    /**
     * Trains the model on the added data.
     *
     * @param model the name of the model
     * @return the stream of RowResult
     * @throws IOException Signals that an I/O exception has occurred.
     * @throws InterruptedException the interrupted exception
     */
    @Procedure
    public Stream<RowResult> train(@Name("model") String model) throws IOException, InterruptedException 
    {
    	MLModel mlModel = MLModel.getModel(model);
    	mlModel.train();
        return Stream.of(new RowResult("Model '"+model+"' trained."));
    }
    
    /**
     * Predicts values for the given features on the model.
     * Every key value pair in the features Map stands for one column of a feature.
     * <p>
     * The keys of the Map <tt>features</tt> must be the same as the ones of Map <tt>types</tt> from the "create" call.
     * <p>
     * If <tt>period</tt> is set to true, <tt>features</tt> has to look like <code>{start: 20170508, end: 20170515}</code> 
     * where the first date value is the start date and the second one the end date.
     * Including those dates, values are automatically predicted for every date in the time period between these two.
     *
     * @param model the name of the model
     * @param features the features
     * @return the stream of PredictResult
     * @throws InterruptedException 
     * @throws IOException 
     */
    @Procedure
    public Stream<PredictResult> predict(@Name("model") String model, @Name("features") Map<String, List<Object>> features) throws IOException, InterruptedException 
    {
    	List<PredictResult> result = new ArrayList<>();
        MLModel mlModel = MLModel.getModel(model);
        
        Collection<Map<String, Object>> predictions = mlModel.predict(features);
        Iterator<Map<String, Object>> iter = predictions.iterator();
        
        while(iter.hasNext())
        {
            result.add(new PredictResult(iter.next()));
        }
        
        return result.stream();
    }
    
    /**
     * Gives general information about the model and the specific implementation.
     *
     * @param model the name of the model
     * @return the stream of RowResult
     */
    @Procedure
    public Stream<RowResult> info(@Name("model") String model) 
    {
    	MLModel mlModel = MLModel.getModel(model);
    	return Stream.of(new RowResult(mlModel.getSpecials().toString()));
    }
    
    /**
     * Removes the model.
     *
     * @param model the name of the model
     * @return the stream of RowResult
     */
    @Procedure
    public Stream<RowResult> remove(@Name("model") String model) 
    {
        return Stream.of(MLModel.remove(model));
    }
    
    
    /**
     * Used as an object to be returned as a stream to Neo4j and displayed as the result of a stored procedure.
     * Represents a String.
     */
    public static class RowResult 
    {
    	/** The result */
	    public String result;
    		    
	    /**
    	 * Instantiates a new RowResult.
    	 *
    	 * @param result the String
    	 */
    	public RowResult(String result) 
	    {
    		this.result = result;	
	    }
    }
    
    /**
     * Used as an object to be returned as a stream to Neo4j and displayed as the result of a stored procedure.
     * Represents a Map of features and the predicted results.
     */
    public static class PredictResult
    {
    	/** The result */
	    public Map<String, Object> result;
    	
    	/**
	     * Instantiates a new PredictResult.
	     *
	     * @param result the Map
	     */
	    public PredictResult(Map<String, Object> result)
    	{
    		this.result = result;
    	}
    }
}


