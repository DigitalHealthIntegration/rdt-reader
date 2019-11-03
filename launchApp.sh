#follow these steps to activate the environment and start the rdt server
#IMP: you need to initialize the conda shell before doing this else this step may fail.

exec bash
conda activate rdt-reader
python3 /home/rdtreader/rdt-reader/django_server/manage.py makemigrations
python3 /home/rdtreader/rdt-reader/django_server/manage.py migrate
python3 /home/rdtreader/rdt-reader/django_server/manage.py runserver 9000

