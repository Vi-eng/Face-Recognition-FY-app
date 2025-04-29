document.getElementById('hmbgr').onclick =  function() {
    dropit();
};


//Functions
function dropit(){
    var dropitems = document.getElementById('nav');
    dropitems.classList.toggle('active');
    var content = document.getElementById('content');
    content.classList.toggle('navactive');
}

function Datetime_format(time){
    var datetimeString = time;
    var datetime = new Date(datetimeString);
    // Extract the components of the date and time
    const year = datetime.getFullYear();      // Full year (e.g., 2024)
    const month = datetime.getMonth() + 1;    // Month, zero-based index (add 1 to get the actual month)
    const day = datetime.getDate();           // Day of the month (1-31)
    const hours = datetime.getHours() - 1;        // Hour (0-23)
    const minutes = datetime.getMinutes();    // Minute (0-59)
    const seconds = datetime.getSeconds();    // Second (0-59)

    // Format the components into a string with the desired format
    const timestamp = `${year}-${month.toString().padStart(2, '0')}-${day.toString().padStart(2, '0')}\ ${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    console.log('Time Stamp', timestamp);
    return timestamp;
};

var title = document.getElementsByTagName('title');
var navs = document.getElementsByClassName('nav');

Array.prototype.forEach.call(navs, (value) => {
    if (title[0].innerHTML == value.innerHTML){
        value.classList.add('acti');
    }else{
        value.classList.remove('acti');
    };    
});

Selector = document.getElementById('Select_name');
if (Selector)
{Selector.addEventListener('change', function() {
    //Clear previous porbs and timestamps
    document.querySelector('tbody').innerHTML = 
    `<tr>
        <td></td>
        <td></td>
        <td id = 'display-td'></td>
    </tr>`;
    
    var selectedOption = this.value;
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            if (xhr.status === 200) {
                var person = JSON.parse(xhr.responseText);
                console.log(`Person's times: ${person.times}`);
                var imagePath = `../static/Images%20and%20Icons/Images/${person.name}.jpg` // format the file path to relative 
                console.log(imagePath);

                if (person.probs){
                for (var i = 0; i < person.probs.length; i++) {
                    console.log(`${person.times[i]} is the person.times[i]`)
                    var last_seen = Datetime_format(person.times[i]);
                    if(i == 0){
                        document.querySelector('tbody').innerHTML = 
                        `<tr>
                            <td class>${last_seen}</td>
                            <td>${person.probs[i]}</td>
                            <td rowspan = "0" id = 'display-td'></td>
                        </tr>`;
                    }else{
                        document.querySelector('tbody').innerHTML += 
                        `<tr>
                            <td>${last_seen}</td>
                            <td>${person.probs[i]}</td>
                        </tr>`;
                    }
                };
            };
                
                document.getElementById('display-td').innerHTML = 
                `<div id = 'display-data'>
                <div id = 'half-1'> <img id = 'manage-img' src= ${imagePath} alt= "${person.name}'s picture" > </div>
                <div id = 'half-2'>
                    <span> </br> Name: ${person.name} </span>
                    <span> </br> Gender: ${person.sex} </span>
                    <span> </br> Description: ${person.des}</span>
                </div>
                </div>`;
                console.log(`Person's name: ${person.name}`);
            } else {
                console.error('Error:', xhr.status);
            }
        }
    };
    xhr.open('GET', '/get_personnel_data?selected_option=' + selectedOption, true);
    xhr.send();
});
};

//Get a tag, onclick, make same request as above, make size of name th and tb 0, expand #Dashlist to take that space, then place all the innerhtml needed!!!
//Each onclick first deselects every other. If you were already clicked, it toggles you and undoes the entire previous event.
var A_tags = document.getElementsByClassName('See_more');
var Name_th = document.getElementById('Name_th');
var Name_tds = document.getElementsByClassName('Name_tds');
var dash_List = document.getElementById('DashList');

Array.prototype.forEach.call(A_tags, (value) => {
    //ONCLICK FUNCTION
    value.addEventListener('click', function(){
        if (value.classList.contains('active')){
            value.classList.remove('active');
            Array.prototype.forEach.call(Name_tds, (td) =>{
                td.classList.remove('hidden');
            });
            Name_th.classList.remove('hidden');
            dash_List.classList.remove('active');
        }else{
            //Remove the active class from ALL others
            Array.prototype.forEach.call(A_tags, (value) =>{
                if (value.classList.contains('active')){
                value.classList.remove('active')
            };
            });
            //Add the active class to the one in question
            value.classList.add('active');
            if (!Name_th.classList.contains('hidden')){
            Array.prototype.forEach.call(Name_tds, (td) =>{
                td.classList.add('hidden');
            });
            Name_th.classList.add('hidden');
            dash_List.classList.add('active');
        };
        };    
    
        var datetimeString = value.innerHTML;
        var datetime = new Date(datetimeString);
        // Extract the components of the date and time
        const year = datetime.getFullYear();      // Full year (e.g., 2024)
        const month = datetime.getMonth() + 1;    // Month, zero-based index (add 1 to get the actual month)
        const day = datetime.getDate();           // Day of the month (1-31)
        const hours = datetime.getHours();        // Hour (0-23)
        const minutes = datetime.getMinutes();    // Minute (0-59)
        const seconds = datetime.getSeconds();    // Second (0-59)

        // Format the components into a string with the desired format
        const timestamp = `${year}-${month.toString().padStart(2, '0')}-${day.toString().padStart(2, '0')}\ ${hours.toString().padStart(2, '0')}_${minutes.toString().padStart(2, '0')}_${seconds.toString().padStart(2, '0')}`;
        console.log('Timestamp:', timestamp);

        var index = Array.prototype.indexOf.call(A_tags, value);
        p_name = Name_tds[index].innerHTML;
        p_prob = document.getElementsByClassName('Prob_td')[index].innerHTML;

        if (value.classList.contains('active')){
            dash_List.innerHTML =
            `<img src = '../static/Images\ and\ Icons/Captures/${p_name}\ ${timestamp}.jpg' id = 'log_img'>
            <div>
            <span> Personnel: ${p_name} </span>
            </div>
            <div>
            <span> Probability: ${p_prob} </span>
            </div>`
        }else{
            dash_List.innerHTML = 
            `<div>
            <span> Personnel </span>
            </div>
            <div>
            <span> Probability </span>
            </div>`
            //Return it to its original state
            //I also have to defime my hidden and active classes 
        }
    });
});



// document.getElementById('Select_name').addEventListener('change', function() {
//     var selectedOption = this.value;
//     var xhr = new XMLHttpRequest();
//     xhr.onreadystatechange = function() {
//         if (xhr.readyState === XMLHttpRequest.DONE) {
//             if (xhr.status === 200) {
//                 var person = JSON.parse(xhr.responseText);
//                 var imagePath = `../static/Images%20and%20Icons/Images/${person.name}.jpg` // format the file path to relative 
//                 console.log(imagePath);
//                 document.getElementById('display-data').innerHTML = 
//                 `<div id = 'half-1'> <img id = 'manage-img' src= ${imagePath} alt= "${person.name}'s picture" > </div>
//                 <div id = 'half-2'>
//                     <span> </br> Name: ${person.name} </span>
//                     <span> </br> Gender: ${person.sex} </span>
//                     <span> </br> Description: ${person.des}</span>
//                 </div>`;
//             } else {
//                 console.error('Error:', xhr.status);
//             }
//         }
//     };
//     xhr.open('GET', '/get_logs_data?selected_option=' + selectedOption, true);
//     xhr.send();
// });

