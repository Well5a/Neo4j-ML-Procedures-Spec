package ml;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Result;
import org.neo4j.kernel.impl.proc.Procedures;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.TestGraphDatabaseFactory;

public class Dl4jRNNTest 
{
	private GraphDatabaseService db;

    @Before
    public void setUp() throws Exception 
    {
        db = new TestGraphDatabaseFactory().newImpermanentDatabase();
        Procedures procedures = ((GraphDatabaseAPI) db).getDependencyResolver().resolveDependency(Procedures.class);
        procedures.registerProcedure(ML.class);
        for(int i = 1; i < 181; i++)
        {
        	db.execute("CREATE (n:User {date: '"+i+"', count: '"+(100 + i*50)+"'})");
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
//    	Result result;
//    	
//    	result = db.execute("CALL ml.create('user', {date: 'numeric'}, {alpha: 0.2, epochs: 75, hidden: 10}, null, true, 'dl4j')");
//        System.out.println("createResult.resultAsString() = \n" + result.resultAsString());
//         
//        result = db.execute("MATCH (n:User) CALL ml.add('user', {date: n.date}, n.count) YIELD result RETURN result");
//        System.out.println("addResult.resultAsString() = \n" + result.resultAsString());
//         
//        result = db.execute("CALL ml.info('user')");
//        System.out.println("infoResult.resultAsString() = \n" + result.resultAsString());
//         
//        result = db.execute("CALL ml.train('user')");
//        System.out.println("trainResult.resultAsString() = \n" + result.resultAsString());
//
//        
//        result = db.execute("CALL ml.predict('user', {start: 110, end: 150})");
//        System.out.println("predictResult.resultAsString() = \n" + result.resultAsString());
//        
//        
//        //Remove models:
//        result = db.execute("CALL ml.remove('user')");
//        System.out.println("removeResult.resultAsString() = \n" + result.resultAsString());
    }
}
