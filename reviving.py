import tensorflow as tf 
import numpy as np 
import joblib
import sklearn
import random

def classificate_neurons(neuron_usefulnesses, percentage, WIDTH, DEPTH):

    """percentage means neurons to classify as useful
    """

    neurons_per_layers = [neuron_usefulnesses[x:x+WIDTH] for x in range(0, WIDTH*DEPTH, WIDTH)]
    for i, l in enumerate(neurons_per_layers):
        threshold = np.percentile(l, percentage)
        neurons_per_layers[i] = np.array(l > threshold, dtype = int)

    return(np.array(neurons_per_layers))


def revive_bad_randomized(session, DEPTH, WIDTH, trainables, classifications):
    
    for d in range(DEPTH):

        input_weight_matrix = session.run(trainables[f"dense_{d}/kernel:0"])
        output_weight_matrix = session.run(trainables[f"dense_{d+1}/kernel:0"])
        new_input_layer, new_output_layer = input_weight_matrix.copy(), output_weight_matrix.copy()

        for w in range(WIDTH):
            if classifications[d, w] == 0:
                input_weights = input_weight_matrix[:, w]
                output_weights = output_weight_matrix[w, :]
                good_neurons = np.argwhere(classifications[d] == 1).flatten()
                # original_input_sumweight = 
                new_input_weights = np.random.normal(np.median(input_weights), np.std(input_weights), len(input_weights))
                new_output_weights = np.random.normal(np.median(output_weights), np.std(output_weights), len(output_weights))
                new_input_layer[:, w] = new_input_weights
                new_output_layer[w, :] = new_output_weights

        # print(d)
        input_tensor = session.graph.get_tensor_by_name(f"dense_{d}/kernel:0")
        output_tensor = session.graph.get_tensor_by_name(f"dense_{d+1}/kernel:0")
        session.run(tf.assign(input_tensor, np.array(new_input_layer)))
        session.run(tf.assign(output_tensor, np.array(new_output_layer)))


def revive_clone_good_with_noise(session, DEPTH, WIDTH, trainables, classifications):

    for d in range(DEPTH):
        
        
        input_weight_matrix = session.run(trainables[f"dense_{d}/kernel:0"])
        output_weight_matrix = session.run(trainables[f"dense_{d+1}/kernel:0"])
        new_input_layer, new_output_layer = input_weight_matrix.copy(), output_weight_matrix.copy()
        good_neurons = np.argwhere(classifications[d] == 1).flatten()
    
        for w in range(WIDTH):
            if classifications[d, w] == 0:
                input_weights = input_weight_matrix[:, w]
                output_weights = output_weight_matrix[w, :]
                
                examplary_neuron = random.choice(good_neurons)
                good_neuron_input_weights = input_weight_matrix[:, examplary_neuron]
                good_neuron_output_weights = output_weight_matrix[examplary_neuron, :]
                # original_input_sumweight = 
                new_input_weights = good_neuron_input_weights + np.random.normal(0, np.std(input_weights), len(input_weights))
                new_output_weights = good_neuron_output_weights + np.random.normal(0, np.std(output_weights), len(output_weights))
                new_input_layer[:, w] = new_input_weights
                new_output_layer[w, :] = new_output_weights
                # print(f'replaced {d}:{w} bad neuron with noise + {d}:{examplary_neuron} good neuron')

        # print(d)
        input_tensor = session.graph.get_tensor_by_name(f"dense_{d}/kernel:0")
        output_tensor = session.graph.get_tensor_by_name(f"dense_{d+1}/kernel:0")
        session.run(tf.assign(input_tensor, np.array(new_input_layer)))
        session.run(tf.assign(output_tensor, np.array(new_output_layer)))


def revive_genetic_algorithm_good_plus_bad(session, DEPTH, WIDTH, trainables, classifications):

    for d in range(DEPTH):
        
        input_weight_matrix = session.run(trainables[f"dense_{d}/kernel:0"])
        output_weight_matrix = session.run(trainables[f"dense_{d+1}/kernel:0"])
        new_input_layer, new_output_layer = input_weight_matrix.copy(), output_weight_matrix.copy()
        good_neurons = np.argwhere(classifications[d] == 1).flatten()
    
        for w in range(WIDTH):
            if classifications[d, w] == 0:
                input_weights = input_weight_matrix[:, w]
                output_weights = output_weight_matrix[w, :]
                
                examplary_neuron = random.choice(good_neurons)
                good_neuron_input_weights = input_weight_matrix[:, examplary_neuron]
                good_neuron_output_weights = output_weight_matrix[examplary_neuron, :]
                # original_input_sumweight = 
                new_input_weights = good_neuron_input_weights + input_weights + np.random.normal(0, np.std(input_weights), len(input_weights))
                new_output_weights = good_neuron_output_weights + output_weights + np.random.normal(0, np.std(output_weights), len(output_weights))
                new_input_layer[:, w] = new_input_weights
                new_output_layer[w, :] = new_output_weights
                # print(f'added {d}:{examplary_neuron} to {d}:{w} bad neuron with noise')

        # print(d)
        input_tensor = session.graph.get_tensor_by_name(f"dense_{d}/kernel:0")
        output_tensor = session.graph.get_tensor_by_name(f"dense_{d+1}/kernel:0")
        session.run(tf.assign(input_tensor, np.array(new_input_layer)))
        session.run(tf.assign(output_tensor, np.array(new_output_layer)))


def revive_genetic_algorithm_good_plus_good(session, DEPTH, WIDTH, trainables, classifications):

    for d in range(DEPTH):
        
        input_weight_matrix = session.run(trainables[f"dense_{d}/kernel:0"])
        output_weight_matrix = session.run(trainables[f"dense_{d+1}/kernel:0"])
        new_input_layer, new_output_layer = input_weight_matrix.copy(), output_weight_matrix.copy()
        good_neurons = np.argwhere(classifications[d] == 1).flatten()
    
        for w in range(WIDTH):
            if classifications[d, w] == 0:
                sum_w = np.random.random()
                input_weights = input_weight_matrix[:, w]
                output_weights = output_weight_matrix[w, :]
                
                n1 = random.choice(good_neurons)
                n1_input_weights = input_weight_matrix[:, n1]
                n1_output_weights = output_weight_matrix[n1, :]

                n2 = random.choice(good_neurons)
                n2_input_weights = input_weight_matrix[:, n2]
                n2_output_weights = output_weight_matrix[n2, :]

                # original_input_sumweight = 
                new_input_weights  = sum_w * n1_input_weights + (1 - sum_w) * n2_input_weights + np.random.normal(0, 0.1, len(input_weights))
                new_output_weights = sum_w * n1_output_weights + (1 - sum_w) * n2_output_weights + np.random.normal(0, 0.1, len(output_weights))
                new_input_layer[:, w] = new_input_weights
                new_output_layer[w, :] = new_output_weights
                print(f'replace {d}:{w} with sum of {d}:{n1} and {d}:{n2} with weight {sum_w:.3f}')

        # print(d)
        input_tensor = session.graph.get_tensor_by_name(f"dense_{d}/kernel:0")
        output_tensor = session.graph.get_tensor_by_name(f"dense_{d+1}/kernel:0")
        session.run(tf.assign(input_tensor, np.array(new_input_layer)))
        session.run(tf.assign(output_tensor, np.array(new_output_layer)))


def get_data_and_predict(session, X_devel, y_devel, BATCH_SIZE, DEPTH, WIDTH, 
                        inputs, labels, mask, dummy_mask, reg_losses, evaluate):

    eval_loss, eval_acc, nonzeros, current_activations, current_zs, labels_evaluated =\
                evaluate(X_devel, y_devel, BATCH_SIZE)

    zs_evaluated = np.empty((DEPTH, 1, WIDTH))
    for i in range(0, len(X_devel), BATCH_SIZE):
        current = [session.run([current_zs],
                                feed_dict={inputs: X_devel[i:i+BATCH_SIZE],
                                            labels: y_devel[i:i+BATCH_SIZE],
                                            mask: dummy_mask})]
        current = np.squeeze(np.array(current))
        zs_evaluated = np.concatenate((zs_evaluated, current), axis=1)

    neuron_activations = zs_evaluated[:, 1:, :]
    neuron_data = []

    # making the data structure correct

    for d in range(DEPTH):
        for w in range(WIDTH):
            current_neuron = [d, DEPTH-1-d, w, session.run(reg_losses)[d].tolist()]
            #print(list(neuron_activations[d, :, w]))
            # TODO BIG HACK, DON'T DO, MISSAVED THE REGRESSOR AND NOW IT'S LIKE THIS
            # WILL FIX IT
            current_neuron += list(neuron_activations[d, :, w])[:999]
            neuron_data += [current_neuron]

    # predicting
    reg = joblib.load('Neuron_predictor/sklearn_logreg/best_regressor_sklearn.joblib')
    scaler = joblib.load('Neuron_predictor/sklearn_logreg/nn_activations_scaler_sklearn.joblib')
    # print(np.array(neuron_data).shape)

    neuron_data = scaler.transform(neuron_data)
    usefulness_per_neuron = reg.predict(neuron_data)
    return(usefulness_per_neuron)

#TODO what to do with double-updating? Treat first or last layer differently?