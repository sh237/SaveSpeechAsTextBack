# SaveSpeechAsTextBack

In "SaveSpeechAsTextBack", I created an API for speech recognition using the Google Cloud Speech-to-Text API and an inference API to automatically generate titles using Hugging Face's model.
# DEMO

The record button allows voice recording, and pressing the same button again initiates voice recognition and displays the results.


https://user-images.githubusercontent.com/78858054/208346388-f50dc2a4-745d-4708-b3e1-9011a1ecdf35.mov

It is also possible to play back recorded audio.

# Features



# Requirement

* Django==4.1.4
* djangorestframework==3.14.0
...

# Installation

Install with requirements.txt.

```bash
pip install -r requirements.txt
```

# Usage

The following commands are executed to make it work in localhost.

```bash
python manage.py createsuperuser
python manage.py runserver
```

You can access the admin site by creating a super user.

# Note

The function to save data is not implemented.

# Author

* Shunya Nagashima
* Twitter : -
* Email : syun864297531@gmail.com

# License

"SaveSpeechAsTextBack" is under [MIT license].

Thank you!
