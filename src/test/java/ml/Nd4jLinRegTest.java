package ml;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Result;
import org.neo4j.kernel.impl.proc.Procedures;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.TestGraphDatabaseFactory;

import ml.ML;

public class Nd4jLinRegTest 
{
	private GraphDatabaseService db;

    @Before
    public void setUp() throws Exception 
    {
        db = new TestGraphDatabaseFactory().newImpermanentDatabase();
        Procedures procedures = ((GraphDatabaseAPI) db).getDependencyResolver().resolveDependency(Procedures.class);
        procedures.registerProcedure(ML.class);
        for(int i = 1; i < 10; i++)
        {
        	db.execute("CREATE (n:User {date: '2017050"+i+"', count: '"+(450000d + i * 500)+"'})");
        }
        for(int i = 1; i < 10; i++)
        {
        	db.execute("CREATE (n:Vehicle {date: '2017050"+i+"', count: '"+(300d + i * 10)+"'})");
        }
    }

    @After
    public void tearDown() throws Exception 
    {
        db.shutdown();
    }

    @Test
    public void predict() throws Exception 
    {
    	Result result;
//    	Result dbmsResult = db.execute("CALL dbms.procedures");
//      System.out.println("dbmsResult.resultAsString() = \n" + dbmsResult.resultAsString());
        
//    	result = db.execute("MATCH (n:User) RETURN n");
//      System.out.println("matchResult.resultAsString() = \n" + result.resultAsString());
        
        //Create model user
    	result = db.execute("CALL ml.create('user', {date: 'numeric'}, {alpha: 0.1, iter: 300, theta: [0.0, 0.0]}, null, true, 'nd4j')");
        System.out.println("createResult.resultAsString() = \n" + result.resultAsString());
         
        result = db.execute("MATCH (n:User) CALL ml.add('user', {date: n.date}, n.count) YIELD result RETURN result");
        System.out.println("addResult.resultAsString() = \n" + result.resultAsString());
         
        result = db.execute("CALL ml.info('user')");
        System.out.println("infoResult.resultAsString() = \n" + result.resultAsString());
         
        result = db.execute("CALL ml.train('user')");
        System.out.println("trainResult.resultAsString() = \n" + result.resultAsString());


        
        
        //Create model vehicle
    	result = db.execute("CALL ml.create('vehicle', {date: 'numeric'}, {alpha: 0.1, iter: 300, theta: [0.0, 0.0]}, {touchMbcWorld: 'true'}, true, 'nd4j')");
        System.out.println("createResult.resultAsString() = \n" + result.resultAsString());
        
        result = db.execute("MATCH (n:Vehicle) CALL ml.add('vehicle', {date: n.date}, n.count) YIELD result RETURN result");
        System.out.println("addResult.resultAsString() = \n" + result.resultAsString());
        
        result = db.execute("CALL ml.info('vehicle')");
        System.out.println("infoResult.resultAsString() = \n" + result.resultAsString());
        
        result = db.execute("CALL ml.train('vehicle')");
        System.out.println("trainResult.resultAsString() = \n" + result.resultAsString());
        
        
        //Predict models:
        result = db.execute("CALL ml.predict('user', {start: 20170508, end: 20170515})");
        System.out.println("predictResult.resultAsString() = \n" + result.resultAsString());
                
        result = db.execute("CALL ml.predict('vehicle', {start: 20170510, end: 20170520})");
        System.out.println("predictResult.resultAsString() = \n" + result.resultAsString());
        
        
        //Remove models:
        result = db.execute("CALL ml.remove('user')");
        System.out.println("removeResult.resultAsString() = \n" + result.resultAsString());
         
        result = db.execute("CALL ml.remove('vehicle')");
        System.out.println("removeResult.resultAsString() = \n" + result.resultAsString());
        
        result = db.execute("CALL ml.create('user',  {date: 'numeric'}, {alpha: 0.1, iter: 300, theta: [0.0, 0.0]}, null, true, 'nd4j') "
        					+ "YIELD result AS createresult "
        					+ "MATCH (n:User) "
        					+ "CALL ml.add('user', {date: n.date}, n.count) "
        					+ "YIELD result "
        					+ "WITH collect(distinct result) AS addresult, createresult "
        					+ "CALL ml.info('user') "
        					+ "YIELD result "
        					+ "WITH collect(distinct result) AS inforesult, addresult, createresult "
        					+ "CALL ml.train('user') "
        					+ "YIELD result AS trainresult "
        					+ "CALL ml.predict('user', {start: 20170508, end: 20170515})"
        					+ "YIELD result "
        					+ "WITH collect(distinct result) AS predictresult, trainresult, inforesult, addresult, createresult "
        					+ "CALL ml.remove('user') "
        					+ "YIELD result AS removeresult "
        					+ "RETURN  createresult, addresult, inforesult, trainresult, predictresult, removeresult");
        System.out.println("createResult.resultAsString() = \n" + result.resultAsString());
        
        result = db.execute("CALL ml.create('user',  {date: 'numeric', test: 'class'}, {alpha: 0.1, iter: 300, theta: [0.0, 0.0, 0.0]}, null, false, 'nd4j') "
				+ "YIELD result AS createresult "
				+ "MATCH (n:User) "
				+ "CALL ml.add('user', {date: n.date, test: 'true'}, n.count) "
				+ "YIELD result "
				+ "WITH collect(distinct result) AS addresult, createresult "
				+ "CALL ml.info('user') "
				+ "YIELD result "
				+ "WITH collect(distinct result) AS inforesult, addresult, createresult "
				+ "CALL ml.train('user') "
				+ "YIELD result AS trainresult "
				+ "CALL ml.predict('user', {date: [20170510, 20170511, 20170512], test: ['true', 'true', 'true']})"
				+ "YIELD result "
				+ "WITH collect(distinct result) AS predictresult, trainresult, inforesult, addresult, createresult "
				+ "CALL ml.remove('user') "
				+ "YIELD result AS removeresult "
				+ "RETURN  createresult, addresult, inforesult, trainresult, predictresult, removeresult");
        System.out.println("createResult.resultAsString() = \n" + result.resultAsString());
    }
}
