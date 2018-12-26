class bounding_box {
	constructor(xmin, ymin, xmax, ymax, confidence = None, classes = None){
		this.xmin = xmin;
		this.ymin = ymin;
		this.xmax = xmax;
		this.ymax = ymax;
		
		this.confidence = confidence;
		this.classes = classes;

		this.label = -1;
    this.score = -1;
  }

	get_label() {
		if (this.label == -1)
      this.label = 0; // np.argmax(this.classes); (Face detection only have one class)
		
    return this.label;
  }
	
	get_score() {
		if (this.score == -1)
      this.score = this.classes[this.get_label()];
			
    return this.score;
  }
}

function decodeYOLOv2(output, img_width, img_height) {
  const anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828];
  let netout = output;
	let nb_class = 1;   // face
	let obj_threshold = 0.4;
  let nms_threshold = 0.3;
  let grid_h = 13, grid_w = 13;

  let size = 4 + nb_class + 1;  // 
  let nb_box = 5;
  let boxes = [];

  // decode the output by the network
  for (let i = 0; i < grid_h * grid_w * nb_box; ++i) {
    netout[size * i + 4] = _sigmoid(netout[size * i + 4]);
  }

  // just one class, don need softmax
  // let classes = [];
  // for (let i = 0; i < grid_h * grid_w * nb_box; ++i) {
  //   classes[i] = [netout[size * i + 5]];
  // }
  // classes = _softmax(classes);

  for (let i = 0; i < grid_h * grid_w * nb_box; ++i) {
    let tmp = (netout[size * i + 4] * 1);
    netout[size * i + 5] = tmp > obj_threshold ? tmp : 0;
  }
  for (let row = 0; row < grid_h; ++row){
    for (let col = 0; col < grid_w; ++col) {
      for (let b = 0; b < nb_box; ++b) {
        // from 4th element onwards are confidence and class classes

        // classes = netout[row,col,b,5:]
        let class_i = [netout[390 * row + 30 * col + 6 * b + 5]];
        
        if (class_i > 0) {
          // first 4 elements are x, y, w, and h
          let x = netout[390 * row + 30 * col + 6 * b + 0]; 
          let y = netout[390 * row + 30 * col + 6 * b + 1];
          let w = netout[390 * row + 30 * col + 6 * b + 2];
          let h = netout[390 * row + 30 * col + 6 * b + 3];

          x = (col + _sigmoid(x)) / grid_w;  // center position, unit: image width
          y = (row + _sigmoid(y)) / grid_h;  // center position, unit: image height
          w = anchors[2 * b + 0] * Math.exp(w) / grid_w;   // unit: image width
          h = anchors[2 * b + 1] * Math.exp(h) / grid_h;   // unit: image height
          confidence = netout[390 * row + 30 * col + 6 * b + 4];
          
          box = new bounding_box(x-w/2, y-h/2, x+w/2, y+h/2, confidence, class_i);
          
          boxes.push(box);
        }
      }
    }
  }

  // suppress non-maximal boxes
  let sorted_boxes = [];
  for (let c = 0; c < nb_class; ++c) {
    sorted_boxes = boxes.sort((a, b) => { return (b.classes[c] - a.classes[c]);});
    for (let i = 0; i < sorted_boxes.length; ++i) {
      if (sorted_boxes[i].classes === 0) continue;
      else {
        for (let j = i + 1; j < sorted_boxes.length; ++j) {
          if (bbox_iou(sorted_boxes[i], sorted_boxes[j]) >= nms_threshold) {
            sorted_boxes[j].classes[c] = 0;
          }
        }
      }
    }
  }

  // remove the boxes which are less likely than a obj_threshold
  let true_boxes = [];
  sorted_boxes.forEach(box => {
    if (box.get_score() > obj_threshold) {
      true_boxes.push(box);
    }
  })

  let result = [];
  for (let i = 0; i < true_boxes.length; ++i) {
    if (true_boxes[i].classes[0] === 0) continue;
    let predicted_class = 'face';
    let score = true_boxes[i].score;
    let a = (true_boxes[i].xmax + true_boxes[i].xmin) * img_width / 2;
    let b = (true_boxes[i].ymax + true_boxes[i].ymin) * img_height / 2;
    let c = (true_boxes[i].xmax - true_boxes[i].xmin) * img_width;
    let d = (true_boxes[i].ymax - true_boxes[i].ymin) * img_height;
    result.push([predicted_class, a, b, c, d, score]);
  }
  return result;
}


function bbox_iou(box1, box2) {
	let intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax]);
	let intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax]);
	
	let intersect = intersect_w * intersect_h;

    let w1 = box1.xmax - box1.xmin;
    let h1 = box1.ymax - box1.ymin;
    let w2 = box2.xmax - box2.xmin;
    let h2 = box2.ymax - box2.ymin;
	
	let union = w1 * h1 + w2 * h2 - intersect;
	
    return intersect / union;
}

function _interval_overlap(interval_a, interval_b){
	let [x1, x2] = interval_a;
	let [x3, x4] = interval_b;

	if (x3 < x1) {
		if (x4 < x1)
			return 0
		else
      return Math.min(x2,x4) - x1
    }
	else {
		if (x2 < x3)
			 return 0
		else
			return Math.min(x2,x4) - x3          
    }
}
function _sigmoid(x){
    return 1 / (1 + Math.exp(-x));
}

// function _softmax(arr, t = -100) {
//   const max = Math.max(...arr);
//   const min = Math.min(...arr);//   const d = arr.map((y) => Math.exp(y - max)).reduce((a, b) => a + b);
//   return arr.map((value, index) => { 
//     if ((min - max) < t) value = value / min * t;
//     return Math.exp(value - max) / d;
//   });
// }

// crop
function crop(x, y, w, h, margin, img_width, img_height) {
  let xmin = Math.floor(x - w * margin);
  let xmax = Math.floor(x + w * margin);
  let ymin = Math.floor(y - h * margin);
  let ymax = Math.floor(y + h * margin);

  if (xmin < 0) xmin = 0;
  if (ymin < 0) ymin = 0;
  if (xmax > img_width) xmax = img_width;
  if (ymax > img_height) ymax = img_height;
  return [xmin, xmax, ymin, ymax];
}

function drawOutput(image, canvas, results, img_width, img_height) {
  canvas.width = img_width;
  canvas.height = img_height;
  ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, img_width, img_height);
	for (let i = 0; i < results.length; ++i) {
		// display detected face
		x = Math.floor(results[i][1]);
		y = Math.floor(results[i][2]);
		w = Math.floor(results[i][3] / 2);
		h = Math.floor(results[i][4] / 2);

		if (w < h) w = h;
		else h = w;

    [xmin,xmax,ymin,ymax] = crop(x, y, w, h, 1.0, img_width, img_height);
    ctx.strokeStyle = "blue";
    ctx.lineWidth = 3;
    ctx.strokeRect(xmin, ymin, xmax-xmin, ymax-ymin);
  }

  // used for face alignment
  function drawKeyPoints(canvas, keypoints) {
    ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, 128, 128);
    for (let i = 0; i < 128; i = i + 2) {
      ctx.beginPath();
      ctx.fillStyle = "red";
      ctx.arc(keypoints[i] * 128, keypoints[i+1] * 128, 2, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }
}