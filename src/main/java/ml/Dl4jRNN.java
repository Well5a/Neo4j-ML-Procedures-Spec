package ml;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionSequenceRecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.ListStringSplit;
import org.datavec.api.split.StringSplit;
import org.datavec.api.writable.ArrayWritable;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class Dl4jRNN extends MLModel
{
	/** The boolean that indicates whether a time series shall be predicted or not. */
	final boolean timePeriod;
	
	/**  The hyperparameters for RNN. */
	Map<String, Object> params;
	
	int trainSize;
    int miniBatchSize;
    
    MultiLayerNetwork net;
	
	
	public Dl4jRNN(String name, Map<String, String> types, Map<String, Object> params, Map<String, String> extra, boolean timePeriod) 
	{
		super(name, types, extra);
		
		this.timePeriod 	= timePeriod;
		this.params 		= params;
		this.trainSize 		= 0;
		this.miniBatchSize 	= 20;
		this.net 			= configureNetwork();
	}

	@Override
	protected void train() throws IOException, InterruptedException 
	{
		Collection<Collection<Collection<Writable>>> colTrain 	= new ArrayList<>();
		Collection<Collection<Collection<Writable>>> colPredict = new ArrayList<>();
		this.trainSize = 100;
		
		for(int i = 0; i < trainSize; i++)
		{
			Collection<Collection<Writable>> seqTrain = new ArrayList<>();
			
			for(int j = 0; j < miniBatchSize; j++)
			{
				seqTrain.add(Arrays.<Writable>asList(new DoubleWritable(Double.parseDouble(rows.get(i+j).get(0))), new DoubleWritable(Double.parseDouble(rows.get(i+j).get(1)))));
			}
			colTrain.add(seqTrain);
			
			Collection<Collection<Writable>> seqPredict = new ArrayList<>();
			seqPredict.add(Arrays.<Writable>asList(new DoubleWritable(Double.parseDouble(rows.get(i+miniBatchSize+1).get(0))), new DoubleWritable(Double.parseDouble(rows.get(i+miniBatchSize+1).get(1)))));
			colPredict.add(seqPredict);
		}				
		SequenceRecordReader trainReader 	= new CollectionSequenceRecordReader(colTrain);
		SequenceRecordReader predictReader 	= new CollectionSequenceRecordReader(colPredict);
		DataSetIterator trainDataIter 		= new SequenceRecordReaderDataSetIterator(trainReader, predictReader, miniBatchSize, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
		
        //Normalize the training data
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fitLabel(true);
        normalizer.fit(trainDataIter);              //Collect training data statistics
        trainDataIter.reset();
        trainDataIter.setPreProcessor(normalizer);
        
		int nEpochs = (int) (long) params.get("epochs");
		
		while(trainDataIter.hasNext())
		{
			System.out.println(trainDataIter.next());
		}
		trainDataIter.reset();
		
		 // ----- Train the network -----
        for (int i = 0; i < nEpochs; i++) 
        {
            net.fit(trainDataIter);
            trainDataIter.reset();
            System.out.println("Epoch: "+i+" / "+nEpochs+"\n");
        }
        
        //Init rnnTimeStep with train data
        while (trainDataIter.hasNext()) 
        {
        	net.rnnTimeStep(trainDataIter.next().getFeatureMatrix());
        }
        trainDataIter.reset();
	}

	@Override
	protected Collection<Map<String, Object>> predict(Map<String, List<Object>> features) throws IOException, InterruptedException 
	{
//		int startDate = Integer.parseInt(String.valueOf(features.get("start")));
//		int endDate = Integer.parseInt(String.valueOf(features.get("end")));
//		
//		int nPredictions = endDate-startDate;
//		System.out.println(nPredictions);
//		double[] dates = new double[nPredictions];
//		
//		for(int i = 0; i < nPredictions; i++)
//		{
//			dates[i] = i+startDate;
//		}
//		INDArray featureArray = Nd4j.create(dates);
//		System.out.println(featureArray);
		Map<String, INDArray> state = net.rnnGetPreviousState(0);
//		List<INDArray> predicted = net.feedForward(input, featuresMask, labelsMask);
//		System.out.println(predicted);
		
		return null;
	}

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
		, "Epochs: "				+ params.get("epochs")
		, "Learnrate (alpha): "		+ params.get("alpha")
		, "Hidden layers: "			+ params.get("hidden")
		);
	}
	
	private MultiLayerNetwork configureNetwork()
	{
		double alpha 		= (double) params.get("alpha");
		int nHidden  		= (int) (long) params.get("hidden");
		int numOfVariables 	= nFeatures+1;
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	            .seed(140)
	            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	            .iterations(1)
	            .weightInit(WeightInit.XAVIER)
	            .updater(Updater.NESTEROVS)
	            .learningRate(alpha)
	            .regularization(true)
	            .list()
	            .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numOfVariables).nOut(nHidden)
	                .build())
	            .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
	                .activation(Activation.IDENTITY).nIn(nHidden).nOut(numOfVariables).build())
	            .build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(20));
		
		return net;
	}
}