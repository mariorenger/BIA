function show_row_infor(element, data) {
	for (let [k, v] of Object.entries(data)){
		let row_element = document.createElement('div')
		let label_element = document.createElement('div')
		label_element.style.display = 'inline-block'
		label_element.innerHTML = k + ': '
		

		let infor_element = document.createElement('div')
		infor_element.style.display = 'inline-block'
		infor_element.style.paddingLeft = '10px'
		infor_element.innerHTML = v
		row_element.appendChild(label_element)
		row_element.appendChild(infor_element)
		element.appendChild(row_element)
	}
}

$(document).ready(function () {
	$('#form1').on('submit', function (event) {
		$.ajax({
			data: {
				AgeMin: $('#AgeMin').val(),
				AgeMax: $('#AgeMax').val(),
				InterestMin: $('#InterestMin').val(),
				InterestMax: $('#InterestMax').val(),
				ImpressionsMin: $('#ImpressionsMin').val(),
				ImpressionsMax: $('#ImpressionsMax').val(),
				ClickMin: $('#ClickMin').val(),
				ClickMax: $('#ClickMax').val(),
				SpentMin: $('#SpentMin').val(),
				SpentMax: $('#SpentMax').val(),
			},
			type: 'POST',
			url: '/process'
		})
			.done(function (data) {
				if (data.error) {
					$('#errorAlert').text(data.error).show();
					$('#successAlert').hide();
				}
				else {
					// $('#successAlert').text(data.total).show();
					var main_element = document.getElementById('successAlert')
					main_element.style.display = '';
					show_row_infor(main_element, data.data);
					main_element.style.fontSize = '16px'
					main_element.show();
					$('#errorAlert').hide();
				}

			});

		event.preventDefault();

	});

});

$(document).ready(function () {
	$('#form2').on('submit', function (event) {
		$.ajax({
			data: {
				Age: $('#Age').val(),
				Gender: $('#Gender').val(),
				Interest: $('#Interest').val(),
				Impressions: $('#Impressions').val(),
				Click: $('#Click').val(),
				Spent: $('#Spent').val(),
			},
			type: 'POST',
			url: 'predict'
		})
			.done(function (data) {

				if (data.error) {
					$('#errorAlert2').text(data.error).show();
					$('#successAlert2').hide();
				}
				else {
					// $('#successAlert2').text(data.total).show();
					var main_element = document.getElementById('successAlert2')
					main_element.style.display = '';
					show_row_infor(main_element, data.data);
					main_element.style.fontSize = '16px'
					main_element.show();
					$('#errorAlert2').hide();
				}

			});

		event.preventDefault();

	});

});