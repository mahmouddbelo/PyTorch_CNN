$(document).ready(function () {
    // Cache DOM elements
    const $imageSection = $('.image-section');
    const $loader = $('.loader');
    const $result = $('#result');
    const $btnPredict = $('#btn-predict');
    const $imagePreview = $('#imagePreview');
    const $uploadFile = $('#upload-file');

    // Initialize
    $imageSection.hide();
    $loader.hide();
    $result.hide();

    // Image preview handler
    function readURL(input) {
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            
            reader.onload = function (e) {
                $imagePreview
                    .css('background-image', `url(${e.target.result})`)
                    .hide()
                    .fadeIn(650);
                
                // Add zoom effect on hover
                $imagePreview.parent().hover(
                    function() { $(this).find('div').css('transform', 'scale(1.1)'); },
                    function() { $(this).find('div').css('transform', 'scale(1.0)'); }
                );
            };
            
            reader.readAsDataURL(input.files[0]);
        }
    }

    // File upload handler
    $("#imageUpload").change(function () {
        // Reset previous results
        $result.hide().removeClass('alert-success alert-danger');
        
        // Show preview section with animation
        $imageSection.fadeIn(500);
        $btnPredict.show();
        
        readURL(this);
    });

    // Prediction handler
    $btnPredict.click(function () {
        const formData = new FormData($uploadFile[0]);

        // UI feedback
        $(this).prop('disabled', true).html('<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...');
        $loader.show();

        // API call
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: formData,
            contentType: false,
            cache: false,
            processData: false,
            success: function (data) {
                // Show result with animation
                $loader.fadeOut(300, function() {
                    $result
                        .removeClass('alert-danger')
                        .addClass('alert-success')
                        .html(`<i class="fas fa-check-circle me-2"></i>Result: ${data}`)
                        .fadeIn(600);
                });
                console.log('Prediction successful!');
            },
            error: function(xhr, status, error) {
                // Handle errors
                $loader.fadeOut(300, function() {
                    $result
                        .removeClass('alert-success')
                        .addClass('alert-danger')
                        .html(`<i class="fas fa-exclamation-circle me-2"></i>Error: ${error}`)
                        .fadeIn(600);
                });
                console.error('Prediction failed:', error);
            },
            complete: function() {
                // Reset button state
                $btnPredict
                    .prop('disabled', false)
                    .html('<i class="fas fa-microscope me-2"></i>Analyze');
            }
        });
    });
});