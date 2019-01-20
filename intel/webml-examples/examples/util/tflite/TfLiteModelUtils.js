function printTfLiteModel(model) {
  function printOperatorCode(operatorCode, i) {
    console.log(`\t operator_codes[${i}]: {builtin_code: ${tflite.BuiltinOperator[operatorCode.builtinCode()]}, custom_code: ${operatorCode.customCode()}}`);
  }
  function printTensor(tensor, i) {
    console.log(`\t\t tensors[${i}]: `+
      `{name: ${tensor.name()}, type: ${tflite.TensorType[tensor.type()]}, shape: [${tensor.shapeArray()}], buffer: ${tensor.buffer()}}`)
  }
  function printOperator(operator, i) {
    let op = tflite.BuiltinOperator[model.operatorCodes(operator.opcodeIndex()).builtinCode()];
    console.log(`\t\t operators[${i}]: `);
    console.log(`\t\t\t {opcode: ${op}, inputs: [${operator.inputsArray()}], outputs: [${operator.outputsArray()}], `)
    switch(op) {
      case 'ADD': {
        let options = operator.builtinOptions(new tflite.AddOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `fused_activation_function: ${tflite.ActivationFunctionType[options.fusedActivationFunction()]}}}`);
      } break;
      case 'MUL': {
        let options = operator.builtinOptions(new tflite.MulOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `fused_activation_function: ${tflite.ActivationFunctionType[options.fusedActivationFunction()]}}}`);
      } break;
      case 'CONV_2D': {
        let options = operator.builtinOptions(new tflite.Conv2DOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `padding: ${tflite.Padding[options.padding()]}, ` +
          `stride_w: ${options.strideW()}, ` +
          `stride_h: ${options.strideH()}, ` +
          `dilation_w: ${options.dilationWFactor()}, `+
          `dilation_h: ${options.dilationHFactor()}, ` +
          `fused_activation_function: ${tflite.ActivationFunctionType[options.fusedActivationFunction()]}}}`);
      } break;
      case 'DEPTHWISE_CONV_2D': {
        let options = operator.builtinOptions(new tflite.DepthwiseConv2DOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `padding: ${tflite.Padding[options.padding()]}, ` +
          `stride_w: ${options.strideW()}, `+
          `stride_h: ${options.strideH()}, ` +
          `dilation_w: ${options.dilationWFactor()}, `+
          `dilation_h: ${options.dilationHFactor()}, ` +
          `depth_multiplier: ${options.depthMultiplier()}, ` +
          `fused_activation_function: ${tflite.ActivationFunctionType[options.fusedActivationFunction()]}}}`);
      } break;
      case 'AVERAGE_POOL_2D': {
        let options = operator.builtinOptions(new tflite.Pool2DOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `padding: ${tflite.Padding[options.padding()]}, ` +
          `stride_w: ${options.strideW()}, ` +
          `stride_h: ${options.strideH()}, ` +
          `filter_width: ${options.filterWidth()}, ` +
          `filter_height: ${options.filterHeight()}, ` +
          `fused_activation_function: ${tflite.ActivationFunctionType[options.fusedActivationFunction()]}}}`);
      } break;
      case 'SOFTMAX': {
        let options = operator.builtinOptions(new tflite.SoftmaxOptions());
        console.log(`\t\t\t  builtin_options: {beta: ${options.beta()}}}`);
      } break;
      case 'RESHAPE': {
        let options = operator.builtinOptions(new tflite.ReshapeOptions());
        console.log(`\t\t\t  builtin_options: {new_shape: [${options.newShapeArray()}]}}`);
      } break;
      case 'MAX_POOL_2D': {
        let options = operator.builtinOptions(new tflite.Pool2DOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `padding: ${tflite.Padding[options.padding()]}, ` +
          `stride_w: ${options.strideW()}, ` +
          `stride_h: ${options.strideH()}, ` +
          `filter_width: ${options.filterWidth()}, ` +
          `filter_height: ${options.filterHeight()}, ` +
          `fused_activation_function: ${tflite.ActivationFunctionType[options.fusedActivationFunction()]}}}`);
      } break;
      case 'CONCATENATION': {
        let options = operator.builtinOptions(new tflite.ConcatenationOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `axis: ${options.axis()}, ` +
          `fused_activation_function: ${tflite.ActivationFunctionType[options.fusedActivationFunction()]}}}`);
      } break;
      case 'SQUEEZE': {
        let options = operator.builtinOptions(new tflite.SqueezeOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `squeezeDims: ${options.squeezeDims()}, ` +
          `squeezeDimsLength: ${options.squeezeDimsLength()}, ` +
          `squeezeDimsArray: ${options.squeezeDimsArray()}}}`);
      } break;
      case 'FULLY_CONNECTED': {
        let options = operator.builtinOptions(new tflite.FullyConnectedOptions());
        console.log(`\t\t\t  builtin_options: {fused_activation_function: ${tflite.ActivationFunctionType[options.fusedActivationFunction()]}}}`);
      } break;
      case 'RESIZE_BILINEAR': {
      } break;
      case 'MAXIMUM': {
      } break;
      default: {
        console.warn(`\t\t\t  builtin_options: ${op} is not supported.}`);
      }
    }
  }
  function printSubgraph(subgraph, i) {
    console.log(`  subgraphs[${i}]`);
    console.log(`\t name: ${subgraph.name()}`);
    console.log(`\t inputs: [${subgraph.inputsArray()}]`);
    console.log(`\t outputs: [${subgraph.outputsArray()}]`);
    console.log(`\t tensors(${subgraph.tensorsLength()}):`)
    for (let i = 0; i < subgraph.tensorsLength(); ++i) {
      printTensor(subgraph.tensors(i), i);
    }
    console.log(`\t operators(${subgraph.operatorsLength()}):`)
    for (let i = 0; i < subgraph.operatorsLength(); ++i) {
      printOperator(subgraph.operators(i), i);
    }
  }
  function printBuffer(buffer, i) {
    console.log(`\t buffer[${i}]: {data: ${buffer.data()}, length: ${buffer.dataLength()}}`);
  }
  console.log(`version: ${model.version()}`);
  console.log(`description: ${model.description()}`);
  console.log(`operator_codes(${model.operatorCodesLength()}):`);
  for (let i = 0; i < model.operatorCodesLength(); ++i) {
    printOperatorCode(model.operatorCodes(i), i);
  }
  console.log(`subgraphs(${model.subgraphsLength()}):`);
  for (let i = 0; i < model.subgraphsLength(); ++i) {
    printSubgraph(model.subgraphs(i), i);
  }
  console.log(`buffers[${model.buffersLength()}]:`);
  for (let i = 0; i < model.buffersLength(); ++i) {
    printBuffer(model.buffers(i), i);
  }
}