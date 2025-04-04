const socket = io();

for (let i = 0; i < 10; i++) {
    const resultDigit = document.createElement('div')
    const resultBar = document.createElement('span')
    const resultProgress = document.createElement('span')

    resultDigit.setAttribute('id', 'result_' + i)
    resultDigit.className = 'result_digit'
    resultDigit.innerText = `${i} `;

    resultBar.className = 'result_bar';
    resultProgress.className = 'result_progress';
    resultBar.appendChild(resultProgress);
    resultDigit.appendChild(resultBar);

    document.getElementById('result').appendChild(resultDigit)
}

socket.on('image_data', (data) => {
    const myElement = document.getElementById("result");
    let index = 0;
    let max = 0;
    let maxIndex = -1;
    for (const child of myElement.children) {
        if (data[index] > max) max = data[index], maxIndex = index;
        child.style = `--value: ${data[index++]}`;
    }
    document.getElementById('number').innerText = max > 0.1 ? String(maxIndex) : '';
})

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
ctx.imageSmoothingEnabled = false;
ctx.strokeWidth = 4;
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
let drawing = false;

const draw = (x, y, width) => {
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, width);
    gradient.addColorStop(0, 'rgba(1.0, 1.0, 1.0, 1.0)');
    gradient.addColorStop(1, 'rgba(1.0, 1.0, 1.0, 0.0)');

    ctx.beginPath();
    ctx.arc(x, y, width, 0, 2 * Math.PI);
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.closePath();
};

canvas.addEventListener('mousedown', e => {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});
canvas.addEventListener('mousemove', e => {
    if (drawing) {
        draw(e.offsetX, e.offsetY, 2);
        sendImage()
    }
});
canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mouseleave', () => drawing = false);

const sendImage = () => {
    const dataURL = canvas.toDataURL('image/png');
    socket.emit('image_data', dataURL);
}

document.getElementById('clear').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    sendImage();
});
