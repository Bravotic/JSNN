// Neural Connection Class
// =============================================================================

// Constructor of the object used to hold information about our neural
// connections
function NeuralConnection(in_weight, in_delta){
  	this.weight = in_weight;
	this.deltaWeight = in_delta; 
}

// Set our weight to something other than what we constructed it with
NeuralConnection.prototype.set_weight = function(in_weight){
	this.weight = in_weight;
};

// Set our delta weight
NeuralConnection.prototype.set_delta_weight = function (in_dw){
	this.deltaWeight = in_dw;
};

// Return the weight to us
NeuralConnection.prototype.get_weight = function(){
	return this.weight;
};

// Same for delta weight
NeuralConnection.prototype.get_delta_weight = function(){
	return this.deltaWeight;
};

// Neuron Class
// =============================================================================

// Constructor for Neuron
function Neuron(outputs, indx){
    
    // Our output connections stored in an array
    this.output_weights = new Array();
    
    // The value, index, and gradient of THIS neuron
    this.output_value = 0;
    this.index = 0;
    this.gradient = 0;
    
    // Learning modifiers

    // The speed of the learning, closer to zero is slower, closer to one is
    // faster but with more error.
    this.eta = 0.50;
    
    // Alpha closer to zero, will engage in more conservative weight
    // modifications. Closer to one and it will do more radical weight
    // modifications.
    this.alpha = 0.25;

    this.output = 0;

    // Go through our output neurons and create new neural connections with them
 // Create them initially with a weight of [random]
    for(var i = 0; i < outputs; i++){
        this.output_weights.push(new NeuralConnection(Math.random(), Math.random()));
    }
    this.index = indx;
}

// Our transfer function, used to normalize data.  Is usually a sigmoid curve,
// in this case we are using a hyperbolic tangent
// Graph of tanh(x) where "+" is a axis
//
//        +
//        + -----
//  ++++++/++++++
//  ----- + 
//        +
//

Neuron.prototype.transfer_function = function(x){
    return (Math.pow(Math.E, 2 * x) - 1) / (Math.pow(Math.E, 2 * x) + 1);
};

// The derivative of the above transfer function
// the derivative of tanh is roughtly equal to "1-x^2" so we just use that
Neuron.prototype.transfer_function_derivative = function(x){
    return 1.0 - Math.pow(x, 2);
};

// Unused but may be used in the future
Neuron.prototype.sumDOW = function(nextLayer){
    var sum = 0;
    for(var n = 0; n < nextLayer.length - 1; n++){
        sum += this.output_weights[n].get_weight() * nextLayer[n].gradient;
    }
};

// Set our output to something
Neuron.prototype.set_output = function(op){
	// Set our private output value to the value we are given
	   this.output_value = op;
};

Neuron.prototype.feed_forward = function(prevLayer){
    var sum = 0.0;
 // For each layer, add up the sum of the outputs and run it through the
 // transfer function (which normalizes the data), that is then set as our
 // output value

    for(var n = 0; n < prevLayer.length; n++){
        sum += prevLayer[n].get_output() * prevLayer[n].output_weights[this.index].get_weight();
    }

    this.output_value = this.transfer_function(sum);
};

Neuron.prototype.calculate_output_gradients = function(target){
	// Calculate the difference between our target and the output we got
	   var delta = target - this.output_value;
	
	// Find the difference between them
	   this.gradient = delta * this.transfer_function_derivative(this.output_value);
	   
	   //console.log("Updated output gradient to: " + this.gradient);
};

Neuron.prototype.calculate_hidden_gradients = function(nextLayer){
	
	// Sum the next layer
	   var sum = 0;
	   for(var n = 0; n < nextLayer.length - 1; n++){
	       //console.log(nextLayer[n].get_gradient());
	sum += this.output_weights[n].get_weight() * nextLayer[n].get_gradient();
	
	   }

	// Calculate gradients
	   this.gradient = sum * this.transfer_function_derivative(this.output_value);

};

Neuron.prototype.update_input_weights = function(prevLayer){
	
// For all of our previous layers, update their weights
	for(var n = 0; n < prevLayer.length; n++){
		var neu = prevLayer[n];
		
		// Calculate the new delta weight from the old delta weights
		var oldDeltaWeight = neu.output_weights[this.index].get_delta_weight();
		var newDeltaWeight = this.eta * neu.get_output() * this.gradient + this.alpha * oldDeltaWeight;
		
		// Set our weights
		neu.output_weights[this.index].set_delta_weight(newDeltaWeight);
		var tmpWeight = neu.output_weights[this.index].get_weight();
		neu.output_weights[this.index].set_weight(tmpWeight + newDeltaWeight);
	}
};

// Get our output
Neuron.prototype.get_output = function(){
	return this.output_value;
};

// Get our gradient
Neuron.prototype.get_gradient = function(){
	return this.gradient;
};

// Get our output weights
Neuron.prototype.get_output_weights = function(){
	return this.output_weights;
};

Neuron.prototype.export = function(){
	return [this.gradient, this.output_weights, this.output_value, this.index, this.alpha, this.eta];
}


// NNetwork Class
// =============================================================================

// Constructor for the network class
function NNetwork(topology){

    // Define our layers array (will hold our layers of neurons)
    this.layers = new Array();

    // Populate those layers with neurons based on the topology we send it
    for(var i = 0; i < topology.length; i++){
        this.layers.push(new Array());
        
        // If the neuron is the last neuron, we make sure it has no outputs as 
        // these are the neurons we will be reading from.
        var outputs = i == topology.length - 1 ? 0 : topology[i+1];

        // Actually create the neurons, giving them their position in the net as
        // well as their outputs
        for(var n = 0; n <= topology[i]; n++){
            this.layers[this.layers.length - 1].push(new Neuron(outputs, n));
            //console.log("Created a new neuron");
        }
        
        // Set the last neuron of each layer to have its output always as 1.
        // This allows us to have a biased neuron which __should__ help with 
        // getting more accurate answers based on our training model.
        this.layers[this.layers.length - 1][this.layers[i].length - 1].set_output(1.0);

    }
}

// Our function to feed forward data into the neural network
NNetwork.prototype.feed_forward = function(input){

    // Set all of our input values to the values of the input we recieve
    for(var i = 0; i < input.length; i++){
        this.layers[0][i].set_output(input[i]);
    }

    // Go through the layers and send the data through, this should go from the
    // previous layer through the current layer.  Notice we start at 1 since the
    // input layers already have their input set, so we don't need to compute
    // anything for them.
    for(var l = 1; l < this.layers.length; l++){
        var previousLayer = this.layers[l-1];

        // Iterate through all the Neurons stored in the layer, and feed them
        // the entire previous layer since we are fully connected.
        for(var n = 0; n < this.layers[l].length - 1; n++){
            this.layers[l][n].feed_forward(previousLayer);
        }
    }
};

NNetwork.prototype.back_prop = function(target){

    // Start with grabbing our output layer
    var output = this.layers[this.layers.length - 1];
    
    // Calcualte the differences between what the NN got and what we wanted the
    // NN to get.
    for(var n = 0; n < output.length - 1; n++){
        output[n].calculate_output_gradients(target[n]);
    }

    // Go backwards through the network and update values we expected vs values
    // we got.  This allows our NN to "learn".
    for(var l = this.layers.length - 2; l > 0; l--){
        var hiddenLayer = this.layers[l];
        var nextLayer = this.layers[l+1];

        for(var n = 0; n < hiddenLayer.length; n++){
            hiddenLayer[n].calculate_hidden_gradients(nextLayer);
        }
    }

    // Go backwards again and update the inputs to make sure we can reflect
    // these changes.

    for(var l = this.layers.length - 1; l > 0; l--){

        
	var currentLayer = this.layers[l];
        var prevLayer = this.layers[l - 1];

        for(var n = 0; n < currentLayer.length - 1; n++){
            currentLayer[n].update_input_weights(prevLayer);
        }
    }
}

NNetwork.prototype.get_results = function(){
    var results = new Array();

    // Go through our output layer and grab all the outputs, send them into an
    // array so we can deal with them ourselves.
    for(var n = 0; n < this.layers[this.layers.length - 1].length - 1; n++){
        //console.log("Adding" + this.layers[this.layers.length - 1][n].get_output());
        results.push(this.layers[this.layers.length - 1][n].get_output());
    }
    return results;
};

