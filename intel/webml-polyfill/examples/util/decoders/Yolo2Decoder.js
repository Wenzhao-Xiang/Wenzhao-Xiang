// reference from https://github.com/experiencor/keras-yolo2
// https://github.com/experiencor/keras-yolo2/blob/master/LICENSE
class bounding_box {
  constructor(xmin, ymin, xmax, ymax, confidence = None, classes = None) {
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
    if (this.label === -1) {
      let max = 0;
      let index = 0;
      for (let i = 0; i < this.classes.length; ++i) {
        if (this.classes[i] > max) {
          max = this.classes[i];
          index = i;
        }
      }
      this.label = index;
    }

    return this.label;
  }

  get_score() {
    if (this.score === -1)
      this.score = this.classes[this.get_label()];

    return this.score;
  }
}

function decodeYOLOv2(options, output, img_width, img_height, anchors) {
  const {
    nb_class = 80,
    nb_box = 5,
    grid_h = 13,
    grid_w = 13,
    obj_threshold = 0.5,
    nms_threshold = 0.3,
  } = options;

  let size = 4 + 1 + nb_class;  // (x, y, w, h) + confidence + classes

  // decode the output by the network
  for (let i = 0; i < grid_h * grid_w * nb_box; ++i) {
    output[size * i + 4] = _sigmoid(output[size * i + 4]);
  }

  let classes = [];
  let indexes = [];
  for (let i = 0; i < grid_h * grid_w * nb_box; ++i) {
    let classes_i = output.slice((nb_class + 5) * i + 5, (nb_class + 5) * (i + 1));
    classes_i = _softmax(classes_i);
    let isOutputClass = false;
    for (let j = 0; j < nb_class; ++j) {
      let tmp = (output[size * i + 4] * classes_i[j]);
      classes_i[j] = 0;
      if (tmp > obj_threshold) {
        classes_i[j] = tmp;
        isOutputClass = true;
      }
    }
    classes.push(classes_i);
    if (isOutputClass) indexes.push(i);
  }

  // get bounding boxes
  let boxes = [];
  indexes.forEach((index) => {
    let class_i = classes[index];
    let b = index % nb_box;
    let col = (index - b) / nb_box % grid_w;
    let row = ((index - b) / nb_box - col) / grid_w % grid_h;
    let x = output[size * index + 0];
    let y = output[size * index + 1];
    let w = output[size * index + 2];
    let h = output[size * index + 3];
    x = (col + _sigmoid(x)) / grid_w;  // center position, unit: image width
    y = (row + _sigmoid(y)) / grid_h;  // center position, unit: image height
    w = anchors[2 * b + 0] * Math.exp(w) / grid_w;   // unit: image width
    h = anchors[2 * b + 1] * Math.exp(h) / grid_h;   // unit: image height
    confidence = output[size * index + 4];

    box = new bounding_box(x-w/2, y-h/2, x+w/2, y+h/2, confidence, class_i);
    boxes.push(box);
  });

  // suppress non-maximal boxes (NMS)
  let tmp_boxes = [];
  let sorted_boxes = [];
  for (let c = 0; c < nb_class; ++c) {
    for (let i = 0; i < boxes.length; ++i) {
      tmp_boxes[i] = [boxes[i], i];
    }
    sorted_boxes = tmp_boxes.sort((a, b) => { return (b[0].classes[c] - a[0].classes[c]);});
    for (let i = 0; i < sorted_boxes.length; ++i) {
      if (sorted_boxes[i][0].classes[c] === 0) continue;
      else {
        for (let j = i + 1; j < sorted_boxes.length; ++j) {
          if (bbox_iou(sorted_boxes[i][0], sorted_boxes[j][0]) >= nms_threshold) {
            boxes[sorted_boxes[j][1]].classes[c] = 0;
          }
        }
      }
    }
  }

  // remove the boxes which are less likely than a obj_threshold
  let true_boxes = [];
  boxes.forEach(box => {
    if (box.get_score() > obj_threshold) {
      true_boxes.push(box);
    }
  });

  let result = [];
  for (let i = 0; i < true_boxes.length; ++i) {
    if (Math.max(...true_boxes[i].classes) === 0) continue;
    let predicted_class_id = true_boxes[i].get_label();
    let score = true_boxes[i].score;
    let a = (true_boxes[i].xmax + true_boxes[i].xmin) * img_width / 2;
    let b = (true_boxes[i].ymax + true_boxes[i].ymin) * img_height / 2;
    let c = (true_boxes[i].xmax - true_boxes[i].xmin) * img_width;
    let d = (true_boxes[i].ymax - true_boxes[i].ymin) * img_height;
    result.push([predicted_class_id, a, b, c, d, score]);
  }
  return result;
}

function getBoxes(results, img_width, img_height, margin) {
  let object_boxes = [];
  for (let i = 0; i < results.length; ++i) {
    // display detected object
    let class_id = results[i][0];
    let x = Math.floor(results[i][1]);
    let y = Math.floor(results[i][2]);
    let w = Math.floor(results[i][3] / 2);
    let h = Math.floor(results[i][4] / 2);
    let prob = results[i][5];

    // used for output square boxes
    // if (w < h)
    //   w = h;
    // else
    //   h = w;

    [xmin, xmax, ymin, ymax] = crop(x, y, w, h, margin, img_width, img_height);
    object_boxes.push([class_id, xmin, xmax, ymin, ymax, prob]);
  }
  return object_boxes;
}

function drawBoxes(image, canvas, object_boxes, labels) {
  ctx = canvas.getContext('2d');
  // drawImage
  canvas.width = image.width / image.height * canvas.height;
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  // drawBox
  let colors = ['purple', 'blue', 'green', 'yellowgreen', 'red', 'orange'];
  object_boxes.forEach(box => {
    let label = labels[box[0]];
    let xmin = box[1] / image.height * canvas.height;
    let xmax = box[2] / image.height * canvas.height;
    let ymin = box[3] / image.height * canvas.height;
    let ymax = box[4] / image.height * canvas.height;
    let prob = box[5];
    ctx.strokeStyle = colors[box[0] % colors.length];
    ctx.fillStyle = colors[box[0] % colors.length];
    ctx.lineWidth = 3;
    ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
    ctx.font = "20px Arial";
    let text = `${label}: ${prob.toFixed(2)}`;
    let width = ctx.measureText(text).width;
    if (xmin >= 2 && ymin >= parseInt(ctx.font, 10)) {
      ctx.fillRect(xmin - 2, ymin - parseInt(ctx.font, 10), width + 4, parseInt(ctx.font, 10));
      ctx.fillStyle = "white";
      ctx.textAlign = 'start';
      ctx.fillText(text, xmin, ymin - 3);
    } else {
      ctx.fillRect(xmin + 2, ymin, width + 4, parseInt(ctx.font, 10));
      ctx.fillStyle = "white";
      ctx.textAlign = 'start';
      ctx.fillText(text, xmin + 2, ymin + 15);
    }
  });
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

function _interval_overlap(interval_a, interval_b) {
  let [x1, x2] = interval_a;
  let [x3, x4] = interval_b;

  if (x3 < x1) {
    if (x4 < x1)
      return 0;
    else
      return Math.min(x2, x4) - x1;
  } else {
    if (x2 < x3)
      return 0;
    else
      return Math.min(x2, x4) - x3;
  }
}

function _sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function _softmax(arr) {
  const max = Math.max(...arr);
  let sum = 0;
  for (let i = 0; i < arr.length; ++i) {
    sum = Math.exp(arr[i] - max) + sum;
  }
  for (let i = 0; i < arr.length; ++i) {
    arr[i] = Math.exp(arr[i] - max) / sum;
  }
  return arr;
}

// crop image
function crop(x, y, w, h, margin, img_width, img_height) {
  let xmin = Math.floor(x - w * margin[0]);
  let xmax = Math.floor(x + w * margin[1]);
  let ymin = Math.floor(y - h * margin[2]);
  let ymax = Math.floor(y + h * margin[3]);

  if (xmin < 0) xmin = 0;
  if (ymin < 0) ymin = 0;
  if (xmax > img_width) xmax = img_width;
  if (ymax > img_height) ymax = img_height;
  return [xmin, xmax, ymin, ymax];
}