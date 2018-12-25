import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.IOException;
import java.util.*;

/*
 * Main class of the TFIDF MapReduce implementation.
 * Author: Tyler Stocksdale
 * Date:   10/18/2017
 */
public class TFIDF {

    public static void main(String[] args) throws Exception {
        // Check for correct usage
        if (args.length != 1) {
            System.err.println("Usage: TFIDF <input dir>");
            System.exit(1);
        }
		
		// Create configuration
		Configuration conf = new Configuration();
		
		// Input and output paths for each job
		Path inputPath = new Path(args[0]);
		Path wcInputPath = inputPath;
		Path wcOutputPath = new Path("output/WordCount");
		Path dsInputPath = wcOutputPath;
		Path dsOutputPath = new Path("output/DocSize");
		Path tfidfInputPath = dsOutputPath;
		Path tfidfOutputPath = new Path("output/TFIDF");
		
		// Get/set the number of documents (to be used in the TFIDF MapReduce job)
   	FileSystem fs = inputPath.getFileSystem(conf);
     	FileStatus[] stat = fs.listStatus(inputPath);
		String numDocs = String.valueOf(stat.length);
		conf.set("numDocs", numDocs);
		
		// Delete output paths if they exist
		FileSystem hdfs = FileSystem.get(conf);
		if (hdfs.exists(wcOutputPath))
			hdfs.delete(wcOutputPath, true);
		if (hdfs.exists(dsOutputPath))
			hdfs.delete(dsOutputPath, true);
		if (hdfs.exists(tfidfOutputPath))
			hdfs.delete(tfidfOutputPath, true);
		
		// Create and execute Word Count job
		
			/************ YOUR CODE HERE ************/
			
		Job job = Job.getInstance(conf, "Word Count");
    	job.setJarByClass(TFIDF.class);
    	job.setMapperClass(WCMapper.class);
    	job.setReducerClass(WCReducer.class);
    	job.setOutputKeyClass(Text.class);
    	job.setOutputValueClass(IntWritable.class);
    	FileInputFormat.addInputPath(job, wcInputPath);
    	FileOutputFormat.setOutputPath(job, wcOutputPath);
		job.waitForCompletion(true);
		
		
			
		// Create and execute Document Size job
		
			/************ YOUR CODE HERE ************/
		job = Job.getInstance(conf, "Document Size");
    	job.setJarByClass(TFIDF.class);
    	job.setMapperClass(DSMapper.class);
    	job.setReducerClass(DSReducer.class);
    	job.setOutputKeyClass(Text.class);
    	job.setOutputValueClass(Text.class);
    	FileInputFormat.addInputPath(job, dsInputPath);
    	FileOutputFormat.setOutputPath(job, dsOutputPath);
		job.waitForCompletion(true);
		
		//Create and execute TFIDF job
		job = Job.getInstance(conf, "TFIDF");
    	job.setJarByClass(TFIDF.class);
    	job.setMapperClass(TFIDFMapper.class);
    	job.setReducerClass(TFIDFReducer.class);
    	job.setOutputKeyClass(Text.class);
    	job.setOutputValueClass(Text.class);
    	FileInputFormat.addInputPath(job, tfidfInputPath);
    	FileOutputFormat.setOutputPath(job, tfidfOutputPath);
		job.waitForCompletion(true);
		
			/************ YOUR CODE HERE ************/
		
    }
	
	/*
	 * Creates a (key,value) pair for every word in the document 
	 *
	 * Input:  ( byte offset , contents of one line )
	 * Output: ( (word@document) , 1 )
	 *
	 * word = an individual word in the document
	 * document = the filename of the document
	 */
	public static class WCMapper extends Mapper<Object, Text, Text, IntWritable> {
		
		/************ YOUR CODE HERE ************/
		private final static IntWritable one = new IntWritable(1);
    	private Text wordDoc = new Text();
	
		private FileSplit fileSplit = new FileSplit();
		private String filename = new String();
		private String outputKey = new String();
		
	 	
    	public void map(Object key, Text value, Context context
    							) throws IOException, InterruptedException {
      
			//get filename
			fileSplit = (FileSplit)context.getInputSplit();
			filename = fileSplit.getPath().getName();
			      
			
			//iterate over each word and combine word with filename to 
			//create a output key
     		StringTokenizer itr = new StringTokenizer(value.toString());
     	 	while (itr.hasMoreTokens()) {
				outputKey = itr.nextToken() + "@" + filename;
       		wordDoc.set(outputKey);
        		context.write(wordDoc, one);
      	}
    	}
		
    }

    /*
	 * For each identical key (word@document), reduces the values (1) into a sum (wordCount)
	 *
	 * Input:  ( (word@document) , 1 )
	 * Output: ( (word@document) , wordCount )
	 *
	 * wordCount = number of times word appears in document
	 */
	public static class WCReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
		
		/************ YOUR CODE HERE ************/
		//result stores wordCount for each key
		private IntWritable result = new IntWritable();

		public void reduce(Text key, Iterable<IntWritable> values,
						   Context context
						   ) throws IOException, InterruptedException {
			int sum = 0;
			//iterate over values and add value to result
			for (IntWritable val : values) {
				sum += val.get();
	 		 }
	  		result.set(sum);
	  		context.write(key, result);
		}	
		
    }
	
	/*
	 * Rearranges the (key,value) pairs to have only the document as the key
	 *
	 * Input:  ( (word@document) , wordCount )
	 * Output: ( document , (word=wordCount) )
	 */
	public static class DSMapper extends Mapper<Object, Text, Text, Text> {
		
		/************ YOUR CODE HERE ************/
		//private final static IntWritable one = new IntWritable(1);
    	private Text outputKey = new Text();
    	private Text outputValue = new Text();
		private String [] splittedKey = new String[2];
		private String [] splittedValue = new String[2];

    	public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      		
			//split value into key and value
			splittedValue = value.toString().split("\t", 0);

			//further split key word@document intto word, document
			splittedKey = splittedValue[0].split("@", 0);

			//create a output key document
        	outputKey.set(splittedKey[1]);
			
			//create a output value word=wordcount
        	outputValue.set(splittedKey[0] + "=" + splittedValue[1]);
	
			//write to context
        	context.write(outputKey, outputValue);
    	}	
    }

    /*
	 * For each identical key (document), reduces the values (word=wordCount) into a sum (docSize) 
	 *
	 * Input:  ( document , (word=wordCount) )
	 * Output: ( (word@document) , (wordCount/docSize) )
	 *
	 * docSize = total number of words in the document
	 */
	public static class DSReducer extends Reducer<Text, Text, Text, Text> {
		
		/************ YOUR CODE HERE ************/
		private IntWritable result = new IntWritable();
    	private String outputKey = new String();
    	private String outputValue = new String();
		private String [] splittedKey = new String[2];
		private String [] splittedValue = new String[2];
		private String output = new String();
		private Text finalKey = new Text();
		private Text finalValue = new Text();

		//stores words for each key document
		private ArrayList<String> ar0 = new ArrayList<String>();

		//sotres wordcount for each word in document
		private ArrayList<String> ar1 = new ArrayList<String>();

		public void reduce(Text key, Iterable<Text> values,
						   Context context
						   ) throws IOException, InterruptedException {
			
			//for every document clear lists
			ar0.clear();
			ar1.clear();

			//set docsize sum to 0
			int sum = 0;

			//for each word in document add its wordCount to sum, add each word, wordcount to list
			for (Text val : values) {
				splittedValue = val.toString().split("=", 0);
				ar0.add(splittedValue[0]);
				ar1.add(splittedValue[1]);
				sum += Integer.parseInt(splittedValue[1]);
	 		}
		
			//for each word in document create output key and output value and write it to context
			for (int i=0; i < ar0.size(); i++) {
				outputKey = ar0.get(i) + "@" + key.toString();
				outputValue = ar1.get(i) + "/" + sum;			
				finalKey.set(outputKey);
				finalValue.set(outputValue);
	  			context.write(finalKey, finalValue);
	 		}
		}	
    }
	
	/*
	 * Rearranges the (key,value) pairs to have only the word as the key
	 * 
	 * Input:  ( (word@document) , (wordCount/docSize) )
	 * Output: ( word , (document=wordCount/docSize) )
	 */
	public static class TFIDFMapper extends Mapper<Object, Text, Text, Text> {

		/************ YOUR CODE HERE ************/
    			
    	private Text outputKey = new Text();
    	private Text outputValue = new Text();
		private String [] splittedKey = new String[2];
		private String InputKey = new String();
		private String wordCount = new String();
		private String [] splittedValue = new String[2];

    	public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
	
			
			//convert input pair to output pair and write to context
			splittedValue = value.toString().split("\t", 0);
			splittedKey = splittedValue[0].split("@", 0);
        	outputKey.set(splittedKey[0]);
        	outputValue.set(splittedKey[1] + "=" + splittedValue[1]);
        	context.write(outputKey, outputValue);
    	}	
		
		
    }

    /*
	 * For each identical key (word), reduces the values (document=wordCount/docSize) into a 
	 * the final TFIDF value (TFIDF). Along the way, calculates the total number of documents and 
	 * the number of documents that contain the word.
	 * 
	 * Input:  ( word , (document=wordCount/docSize) )
	 * Output: ( (document@word) , TFIDF )
	 *
	 * numDocs = total number of documents
	 * numDocsWithWord = number of documents containing word
	 * TFIDF = (wordCount/docSize) * ln(numDocs/numDocsWithWord)
	 *
	 * Note: The output (key,value) pairs are sorted using TreeMap ONLY for grading purposes. For
	 *       extremely large datasets, having a for loop iterate through all the (key,value) pairs 
	 *       is highly inefficient!
	 */
	public static class TFIDFReducer extends Reducer<Text, Text, Text, Text> {
		
		private static int numDocs;
		private Map<Text, Text> tfidfMap = new HashMap<Text, Text>();
		private ArrayList<String> documentsVals = new ArrayList<String>();

    	private String outputKey = new String();
    	private String outputValue = new String();
		private String [] splittedKey = new String[2];
		private String [] splittedValue = new String[2];
		private String [] splittedWcDs = new String[2];
		
		// gets the numDocs value and stores it
		protected void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			numDocs = Integer.parseInt(conf.get("numDocs"));
		}
		
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			
			/************ YOUR CODE HERE ************/
			//clear document=wordCount/docSize list when reduce is called for each key
			documentsVals.clear();

			//loop over iterator to convert values to string and also implicitely know
			//how may documents has that word
			for(Text val : values) {
				documentsVals.add(val.toString());
			}
			
			//number of docs having word in key
			int nbDocsWithWord = documentsVals.size();
			
			//word count for each word in document
			int wc;

			//document size
			int ds;
		
			//TFIDF
			double tfidfVal;
			
			double nbDocsByNbDocsWithWord = numDocs * 1.0 / nbDocsWithWord;
			double lnNbDocsByNbDocsWithWord = Math.log(nbDocsByNbDocsWithWord);

			for(String docVal : documentsVals) {
				Text finalKey = new Text();
				Text finalValue = new Text();
				
				//create output key
				splittedValue = docVal.split("=", 0);
				outputKey = splittedValue[0] + "@" + key.toString();		
				
				//calculate TFIDF and create ouput value
				splittedWcDs = splittedValue[1].split("/", 0);
				wc = Integer.parseInt(splittedWcDs[0]);
				ds = Integer.parseInt(splittedWcDs[1]);
				tfidfVal = (wc * 1.0 / ds) * lnNbDocsByNbDocsWithWord;
				outputValue = Double.toString(tfidfVal);
				
				finalKey.set(outputKey);
				finalValue.set(outputValue);
	 
				//Put the output (key,value) pair into the tfidfMap instead of doing a context.write
				tfidfMap.put(finalKey, finalValue);
			}
		}
		
		// sorts the output (key,value) pairs that are contained in the tfidfMap
		protected void cleanup(Context context) throws IOException, InterruptedException {
            Map<Text, Text> sortedMap = new TreeMap<Text, Text>(tfidfMap);
			for (Text key : sortedMap.keySet()) {
                context.write(key, sortedMap.get(key));
            }
        }
		
    }
}
