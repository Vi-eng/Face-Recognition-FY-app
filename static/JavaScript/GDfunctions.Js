document.getElementById('tname').addEventListener('focus', function() {
    var addButton = document.querySelector('button');
    addButton.classList.toggle('translateY');
    document.getElementById('namefd').classList.toggle('exlength');
    setInterval(function (){bluradd();}, 100 );
});

document.getElementById('tname').addEventListener('blur', function onblur() {
    var runthis = true;
    document.querySelector('button').onclick = function(){
        document.getElementById('tname').removeEventListener('blur', onblur);
        runthis = false;
        console.log(runthis);  
    };
    var addButton = document.querySelector('button');
    addButton.classList.toggle('translateY');
    document.getElementById('namefd').classList.toggle('exlength');
    console.log(runthis);
});

document.getElementById('hmbgr').onclick =  function() {
    dropit();
};

document.getElementById('comp').addEventListener('click', function(){selectComp();});

document.getElementById('incomp').addEventListener('click', function(){selectinComp();});


//Functions
function dropit(){
    var dropitems = document.getElementById('nav');
    dropitems.classList.toggle('active');
    var content = document.getElementById('content');
    content.classList.toggle('navactive');
}

function bluradd(){
var textbox = document.querySelector('textarea').value;
var addButton = document.querySelector('button');
if (textbox != ''){
    addButton.classList.add('opaque');
    addButton.addEventListener('click', function(){addtask(); clear();});
} else if(textbox == '' & addButton.classList.contains('opaque')){
    addButton.classList.toggle('opaque');
};
}

function selectComp(){
    var sincom = document.getElementById('incomp');
    var scom = document.getElementById('comp');

    if(scom.classList.contains('bactive')){
        sincom.classList.toggle('bactive');
        scom.classList.toggle('bactive');
    };
}

function selectinComp(){
    var sincom = document.getElementById('incomp');
    var scom = document.getElementById('comp');

    if(sincom.classList.contains('bactive')){
        scom.classList.toggle('bactive');
        sincom.classList.toggle('bactive');
    };
}

function addtask(){
    var listdiv = document.getElementById('listitems');
    var listit = listdiv.cloneNode(true);
    var li = listit.querySelector('span');
    var left = document.getElementById('left');
    var insertdiv = left.getElementsByClassName('hide')[0];

    var taskname = document.querySelector('textarea').value;
    var listcontent = document.createTextNode(taskname);
    listit.insertBefore(listcontent,li); 
    insertdiv.insertBefore(listit,listdiv); 
    console.log(insertdiv, listit);
}

function clear(){
    var taskname = document.querySelector('textarea').value;
    document.querySelector('textarea').value = '';
}