"""
WSGI config for gis_6ban_1 project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/wsgi/
"""
#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gis_6ban_1.settings')

application = get_wsgi_application()
