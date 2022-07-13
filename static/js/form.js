$(document).ready(function() {

	$('#form1').on('submit', function(event) {

		$.ajax({
			data : {
				AgeMin : $('#AgeMin').val(),
				AgeMax : $('#AgeMax').val(),
				InterestMin : $('#InterestMin').val(),
				InterestMax : $('#InterestMax').val(),
				ImpressionsMin : $('#ImpressionsMin').val(),
				ImpressionsMax : $('#ImpressionsMax').val(),
				ClickMin : $('#ClickMin').val(),
				ClickMax : $('#ClickMax').val(),
				SpentMin : $('#SpentMin').val(),
				SpentMax : $('#SpentMax').val(),
			},
			type : 'POST',
			url : '/process'
		})
		.done(function(data) {

			if (data.error) {
				$('#errorAlert').text(data.error).show();
				$('#successAlert').hide();
			}
			else {
				$('#successAlert').text(data.total).show();
				$('#errorAlert').hide();
			}

		});

		event.preventDefault();

	});

});

$(document).ready(function() {

	$('#form2').on('submit', function(event) {

		$.ajax({
			data : {
				Age : $('#Age').val(),
				Gender : $('#Gender').val(),
				Interest : $('#Interest').val(),
				Impressions : $('#Impressions').val(),
				Click : $('#Click').val(),
				Spent : $('#Spent').val(),
			},
			type : 'POST',
			url : 'predict'
		})
		.done(function(data) {

			if (data.error) {
				$('#errorAlert2').text(data.error).show();
				$('#successAlert2').hide();
			}
			else {
				$('#successAlert2').text(data.total).show();
				$('#errorAlert2').hide();
			}

		});

		event.preventDefault();

	});

});