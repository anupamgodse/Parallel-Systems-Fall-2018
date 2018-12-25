import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.*;
import scala.Tuple2;

import java.util.*;

/*
 * Main class of the TFIDF Spark implementation.
 * Author: Tyler Stocksdale
 * Date:   10/31/2017
 */
public class TFIDF {

	static boolean DEBUG = false;

    public static void main(String[] args) throws Exception {
        // Check for correct usage
        if (args.length != 1) {
            System.err.println("Usage: TFIDF <input dir>");
            System.exit(1);
        }
		
		// Create a Java Spark Context
		SparkConf conf = new SparkConf().setAppName("TFIDF");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// Load our input data
		// Output is: ( filePath , fileContents ) for each file in inputPath
		String inputPath = args[0];
		JavaPairRDD<String,String> filesRDD = sc.wholeTextFiles(inputPath);
		
		// Get/set the number of documents (to be used in the IDF job)
		long numDocs = filesRDD.count();
		
		//Print filesRDD contents
		if (DEBUG) {
			List<Tuple2<String, String>> list = filesRDD.collect();
			System.out.println("------Contents of filesRDD------");
			for (Tuple2<String, String> tuple : list) {
				System.out.println("(" + tuple._1 + ") , (" + tuple._2.trim() + ")");
			}
			System.out.println("--------------------------------");
		}
		
		/* 
		 * Initial Job
		 * Creates initial JavaPairRDD from filesRDD
		 * Contains each word@document from the corpus and also attaches the document size for 
		 * later use
		 * 
		 * Input:  ( filePath , fileContents )
		 * Map:    ( (word@document) , docSize )
		 */
		JavaPairRDD<String,Integer> wordsRDD = filesRDD.flatMapToPair(
			new PairFlatMapFunction<Tuple2<String,String>,String,Integer>() {
				public Iterable<Tuple2<String,Integer>> call(Tuple2<String,String> x) {
					// Collect data attributes
					String[] filePath = x._1.split("/");
					String document = filePath[filePath.length-1];
					String fileContents = x._2;
					String[] words = fileContents.split("\\s+");
					int docSize = words.length;
					
					// Output to Arraylist
					ArrayList ret = new ArrayList();
					for(String word : words) {
						ret.add(new Tuple2(word.trim() + "@" + document, docSize));
					}
					return ret;
				}
			}
		);
		
		//Print wordsRDD contents
		if (DEBUG) {
			List<Tuple2<String, Integer>> list = wordsRDD.collect();
			System.out.println("------Contents of wordsRDD------");
			for (Tuple2<String, Integer> tuple : list) {
				System.out.println("(" + tuple._1 + ") , (" + tuple._2 + ")");
			}
			System.out.println("--------------------------------");
		}		
		
		/* 
		 * TF Job (Word Count Job + Document Size Job)
		 * Gathers all data needed for TF calculation from wordsRDD
		 *
		 * Input:  ( (word@document) , docSize )
		 * Map:    ( (word@document) , (1/docSize) )
		 * Reduce: ( (word@document) , (wordCount/docSize) )
		 */
		JavaPairRDD<String,String> tfRDD = wordsRDD.mapToPair(
			new PairFunction<Tuple2<String,Integer>,String,String>() {
				public Tuple2<String,String> call(Tuple2<String,Integer> x) {
					
					//create output value
					String inverse = "1/" + x._2; 
					
					//return output tuple	
					return new Tuple2<String, String>(x._1, inverse);
				}
			}
			
		).reduceByKey(
			new Function2<String, String, String>() {
				public String call(String x, String y) {
					
					String[] x_split = x.split("/");
					String[] y_split = y.split("/");
					
					//add numerators to get wordCount
					int wordCount = Integer.parseInt(x_split[0]) + Integer.parseInt(y_split[0]);
				
					//create output value
					String outputValue = wordCount + "/" + x_split[1];					

					
					return outputValue;
				}
			}
			
		);
		
		//Print tfRDD contents
		if (DEBUG) {
			List<Tuple2<String, String>> list = tfRDD.collect();
			System.out.println("-------Contents of tfRDD--------");
			for (Tuple2<String, String> tuple : list) {
				System.out.println("(" + tuple._1 + ") , (" + tuple._2 + ")");
			}
			System.out.println("--------------------------------");
		}
		
		/*
		 * IDF Job
		 * Gathers all data needed for IDF calculation from tfRDD
		 *
		 * Input:  ( (word@document) , (wordCount/docSize) )
		 * Map:    ( word , (1/document) )
		 * Reduce: ( word , (numDocsWithWord/document1,document2...) )
		 * Map:    ( (word@document) , (numDocs/numDocsWithWord) )
		 */
		JavaPairRDD<String,String> idfRDD = tfRDD.mapToPair(
			
			new PairFunction<Tuple2<String,String>,String,String>() {
				public Tuple2<String,String> call(Tuple2<String,String> x) {
					
					//seperate word and document in word@document
					String[] x_1_split = x._1.split("@");
					
					//return output tuple	
					return new Tuple2<String, String>(x_1_split[0], 1 + "/" + x_1_split[1]);
				}
			}
			
			
		).reduceByKey(
			new Function2<String, String, String>() {
				public String call(String x, String y) {
					
					String[] x_split = x.split("/");
					String[] y_split = y.split("/");
					
					//combine documents by , to make document list containing key word
					String docList = x_split[1] + "," + y_split[1];

					//add numerators to get docCount
					int numDocsWithWord = Integer.parseInt(x_split[0]) + Integer.parseInt(y_split[0]);
					
					//create output value
					String outputValue = numDocsWithWord + "/" + docList;					

					
					return outputValue;
				}
			}
		).flatMapToPair(
						
			new PairFlatMapFunction<Tuple2<String,String>,String,String>() {
				public Iterable<Tuple2<String,String>> call(Tuple2<String,String> x) {
					
					//seperate word and document in word@document
					String[] x_2_split = x._2.split("/");
			
					String numDocsWithWord = x_2_split[0];

					String[] documents = x_2_split[1].split(",");

					String word = x._1;
					
					//output to arraylist
					ArrayList ret = new ArrayList();
					for(String doc : documents) {
						ret.add(new Tuple2(word + "@" + doc, numDocs + "/" + numDocsWithWord));
					}
					return ret;
					
				}
			}
			
		);
		
		//Print idfRDD contents
		if (DEBUG) {
			List<Tuple2<String, String>> list = idfRDD.collect();
			System.out.println("-------Contents of idfRDD-------");
			for (Tuple2<String, String> tuple : list) {
				System.out.println("(" + tuple._1 + ") , (" + tuple._2 + ")");
			}
			System.out.println("--------------------------------");
		}
		
		/*
		 * TF * IDF Job
		 * Calculates final TFIDF value from tfRDD and idfRDD
		 *
		 * Input:  ( (word@document) , (wordCount/docSize) )          [from tfRDD]
		 * Map:    ( (word@document) , TF )
		 * 
		 * Input:  ( (word@document) , (numDocs/numDocsWithWord) )    [from idfRDD]
		 * Map:    ( (word@document) , IDF )
		 * 
		 * Union:  ( (word@document) , TF )  U  ( (word@document) , IDF )
		 * Reduce: ( (word@document) , TFIDF )
		 * Map:    ( (document@word) , TFIDF )
		 *
		 * where TF    = wordCount/docSize
		 * where IDF   = ln(numDocs/numDocsWithWord)
		 * where TFIDF = TF * IDF
		 */
		JavaPairRDD<String,Double> tfFinalRDD = tfRDD.mapToPair(
			new PairFunction<Tuple2<String,String>,String,Double>() {
				public Tuple2<String,Double> call(Tuple2<String,String> x) {
					double wordCount = Double.parseDouble(x._2.split("/")[0]);
					double docSize = Double.parseDouble(x._2.split("/")[1]);
					double TF = wordCount/docSize;
					return new Tuple2(x._1, TF);
				}
			}
		);
		
		JavaPairRDD<String,Double> idfFinalRDD = idfRDD.mapToPair(
			new PairFunction<Tuple2<String,String>,String,Double>() {
				public Tuple2<String,Double> call(Tuple2<String,String> x) {
					double numDocsWithWord = Double.parseDouble(x._2.split("/")[1]);
					double IDF = Math.log(numDocs/numDocsWithWord);
					return new Tuple2(x._1, IDF);
				}
			}
			
		);
		
		JavaPairRDD<String,Double> tfidfRDD = tfFinalRDD.union(idfFinalRDD).reduceByKey(
			new Function2<Double, Double, Double>() {
				public Double call(Double x, Double y) {
					return x*y;
				}
			}
			
		).mapToPair(
			new PairFunction<Tuple2<String,Double>,String,Double>() {
				public Tuple2<String,Double> call(Tuple2<String,Double> x) {
					String[] x_1_split = x._1.split("@");
					String word = x_1_split[0];
					String document = x_1_split[1];
					return new Tuple2(document + "@" + word,  x._2);
				}
			}
			
		);
		
		//Print tfidfRDD contents in sorted order
		Map<String, Double> sortedMap = new TreeMap<>();
		List<Tuple2<String, Double>> list = tfidfRDD.collect();
		for (Tuple2<String, Double> tuple : list) {
			sortedMap.put(tuple._1, tuple._2);
		}
		if(DEBUG) System.out.println("-------Contents of tfidfRDD-------");
		for (String key : sortedMap.keySet()) {
			System.out.println(key + "\t" + sortedMap.get(key));
		}
		if(DEBUG) System.out.println("--------------------------------");
	}	
}
