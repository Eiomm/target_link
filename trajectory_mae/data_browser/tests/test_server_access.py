"""A user may follow a homepage link without granting cross-site data access."""
import json
from email.message import Message
from types import SimpleNamespace

import pytest

from trajectory_mae.data_browser.server import handler_for


@pytest.mark.parametrize('path,mode,dest,status',[
    ('/','navigate','document',200),
    ('/index.html','navigate','document',200),
    ('/?from=chat','navigate','document',200),
    ('/','navigate','iframe',403),
    ('/','cors','empty',403),
    ('/',None,None,403),
    ('/api/overview','navigate','document',403),
    ('/api/overview','cors','empty',403),
    ('/app.js','no-cors','script',403),
])
def test_cross_site_homepage_navigation(path,mode,dest,status):
    fake=SimpleNamespace(**{name:lambda **kwargs: {} for name in ['overview','cells','locate','cell','group']})
    handler=object.__new__(handler_for(fake))
    handler.path=path
    handler.server=SimpleNamespace(server_port=8765)
    handler.headers=Message()
    handler.headers['Host']='127.0.0.1:8765'
    handler.headers['Sec-Fetch-Site']='cross-site'
    if mode:handler.headers['Sec-Fetch-Mode']=mode
    if dest:handler.headers['Sec-Fetch-Dest']=dest
    response=[]
    handler.send=lambda code,body,*args:response.append((code,body))
    handler.do_GET()
    assert response[0][0]==status
    if status==200:assert b'<html' in response[0][1]


@pytest.mark.parametrize('site',['same-origin','none',None])
def test_normal_local_api_access(site):
    fake=SimpleNamespace(**{name:lambda **kwargs: {'ok':True} for name in ['overview','cells','locate','cell','group']})
    handler=object.__new__(handler_for(fake))
    handler.path='/api/overview'
    handler.server=SimpleNamespace(server_port=8765)
    handler.headers=Message()
    handler.headers['Host']='localhost:8765'
    if site:handler.headers['Sec-Fetch-Site']=site
    response=[]
    handler.send=lambda code,body,*args:response.append((code,body))
    handler.do_GET()
    assert response[0][0]==200 and json.loads(response[0][1])=={'ok':True}


@pytest.mark.parametrize('host',[
    'localhost:54454','127.0.0.1:54454','[::1]:54454',
    'localhost','127.0.0.1','[::1]','LOCALHOST:8765',
])
@pytest.mark.parametrize('path',['/','/app.js','/api/overview'])
def test_forwarded_loopback_ports(host,path):
    fake=SimpleNamespace(**{name:lambda **kwargs: {'ok':True} for name in ['overview','cells','locate','cell','group']})
    handler=object.__new__(handler_for(fake))
    handler.path=path
    handler.server=SimpleNamespace(server_port=8765)
    handler.headers=Message()
    handler.headers['Host']=host
    handler.headers['Sec-Fetch-Site']='same-origin'
    response=[]
    handler.send=lambda code,body,*args:response.append((code,body))
    handler.do_GET()
    assert response[0][0]==200


@pytest.mark.parametrize('host',[
    '', 'evil.example:54454', 'localhost.evil.example:54454',
    'localhost@evil.example:54454','evil.example@localhost:54454',
    'localhost:0','localhost:65536','localhost:',
    'localhost:54454/path','localhost:54454?query', 'localhost:54454#fragment',
    'localhost:54454,evil.example', ' localhost:54454', '127.0.0.1:abc',
])
def test_reject_non_loopback_or_malformed_host(host):
    from trajectory_mae.data_browser.server import is_loopback_host
    assert not is_loopback_host(host)


def test_duplicate_host_is_rejected():
    handler=object.__new__(handler_for(None))
    handler.path='/'
    handler.headers=Message()
    handler.headers['Host']='localhost:54454'
    handler.headers['Host']='evil.example'
    response=[]
    handler.send=lambda code,body,*args:response.append((code,body))
    handler.do_GET()
    assert response[0][0]==403
