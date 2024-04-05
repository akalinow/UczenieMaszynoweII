
import glob 
import tensorflow as tf
from termcolor import colored

#####################################################################
def map_fn(x):
    start = tf.random.uniform(shape=[], maxval = max_window_end-window_size, dtype=tf.int32)
    end = start + window_size
    features =  tf.concat((x[:,start:end], x[:,end+1:end+1+window_size]), axis=1)
    label = x[:,end]

    return features, label
#####################################################################
def print_item(batch, vocabulary):
    batch_index = 0
    item = (batch[0][batch_index], batch[1][batch_index])
    features = " ".join(vocabulary[item[0].numpy()[0:window_size]])
    label = vocabulary[item[1].numpy()]   
    print(colored("Features", "blue"), end=" ")
    print(colored("(Label):", "red"), end=" ")

    print(features, end=" ")
    print(colored(label,"red"), end=" ")
    features = " ".join(vocabulary[item[0].numpy()[window_size:]])
    print(features)
#####################################################################
# Function to read a text file and return a filtered dataset
def load_wksf_dataset(filePath):
    fileList = glob.glob(filePath + "/*.txt")
    if filePath.find(".txt") != -1:
        fileList = [filePath]
    print(colored("Reading text from files:","blue"),fileList)
    dataset = tf.data.TextLineDataset(fileList)
    dataset = dataset.filter(lambda x: not tf.strings.regex_full_match(x, ".*[~].*"))
    dataset = dataset.filter(lambda x: not tf.strings.regex_full_match(x, ".*[<].*"))
    dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "\[[0-9]+\]", "", replace_global=True))
    return dataset
#####################################################################
def load_wikipedia_daset(filePath):

    dataset = tf.keras.preprocessing.text_dataset_from_directory(filePath,
                labels=None, label_mode=None, class_names=None, batch_size=batchSize,
                max_length=None, shuffle=False, seed=None, validation_split=None,
                subset=None, follow_links=False)

    #Remove HTML tags
    regexp = "<[^>]*>"
    dataset = dataset.map(lambda x: tf.strings.regex_replace(x, regexp, "", replace_global=True))
    return dataset
#####################################################################
def preprocess_text(dataset):

    #Vectorize
    vectorize_layer = tf.keras.layers.TextVectorization(output_mode = "int", max_tokens=100000)
    vectorize_layer.adapt(dataset)
    vocabulary = np.array(vectorize_layer.get_vocabulary())
    vocabulary_length = vocabulary.shape[0] 
    dataset = dataset.map(vectorize_layer, num_parallel_calls=tf.data.AUTOTUNE)
    print(colored("Vocabulary length: ", "blue"), vocabulary_length)

    #Remove short texts
    dataset = dataset.filter(lambda x: tf.shape(x)[-1] > max_window_end)

    #Tokenize and optimize I/O
    dataset = dataset.map(map_fn).prefetch(tf.data.AUTOTUNE)

    #Print a few examples
    #Original text
    batch_iter = iter(dataset)
    batch = next(batch_iter)
    print(colored("Original text: ", "blue"), batch[0].numpy().decode("utf-8"))

    #Features and label
    for item in dataset.take(10):
        print_item(item, vocabulary)

    return dataset, vectorize_layer
#####################################################################    
def dump_embedding(model, vocabulary):
  import io
  out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
  out_m = io.open('metadata.tsv', 'w', encoding='utf-8')
  weights = model.get_layer('embedding').get_weights()[0]
  for index, word in enumerate(vocabulary):
    if index == 0:
      continue  # skip 0, it's padding.
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_m.write(word + "\n")
  out_v.close()
  out_m.close()
#####################################################################



