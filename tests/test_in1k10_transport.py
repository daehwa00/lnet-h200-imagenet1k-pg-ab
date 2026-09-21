"""Offline transport backoff and diagnostic regression tests."""
import io
import sys
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from in1k10_transport import API,error_details,retry_delay


def failure(code=429,body=b'provider request limit exceeded',retry_after='120'):
    headers=Message()
    if retry_after is not None:headers['Retry-After']=retry_after
    return urllib.error.HTTPError('https://example.invalid',code,'failure',headers,io.BytesIO(body))


class TransportTests(unittest.TestCase):
    def test_rate_limit_respects_server_delay(self):
        with patch('in1k10_transport.urllib.request.urlopen',side_effect=[failure(),io.BytesIO(b'{"ok": true}')]) as request,patch('in1k10_transport.time.sleep') as sleep:
            self.assertEqual(API('key').call('/control'),{'ok':True})
        self.assertEqual(request.call_count,2)
        sleep.assert_called_once_with(120.)

    def test_single_attempt_never_sleeps_and_keeps_error(self):
        error=failure()
        with patch('in1k10_transport.urllib.request.urlopen',side_effect=error) as request,patch('in1k10_transport.time.sleep') as sleep:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                API('key').call('/control',attempts=1,timeout=8)
        self.assertIs(caught.exception,error)
        self.assertEqual(request.call_count,1)
        self.assertEqual(request.call_args.kwargs['timeout'],8)
        sleep.assert_not_called()
        self.assertEqual(error_details(error)['body_excerpt'],'provider request limit exceeded')
        self.assertEqual(error_details(error)['http_status'],429)

    def test_body_cached_once_and_redacted(self):
        token='short-private-owner-key'
        error=failure(body=('provider limit; '+token+'; Bearer abc; token="hidden-value"; '+'a'*64).encode())
        with patch.object(error,'read',wraps=error.read) as read,patch('in1k10_transport.urllib.request.urlopen',side_effect=error):
            with self.assertRaises(urllib.error.HTTPError):API(token).call('/control',attempts=1)
            details=error_details(error)
            self.assertEqual(details,error_details(error))
            read.assert_called_once_with(4096)
        self.assertIn('provider limit',details['body_excerpt'])
        for secret in (token,'abc','hidden-value','a'*64):self.assertNotIn(secret,details['body_excerpt'])
        self.assertLessEqual(len(details['body_excerpt']),400)

    def test_http_date_and_cap(self):
        with patch('in1k10_transport.time.time',return_value=0):
            self.assertEqual(retry_delay(failure(retry_after='Thu, 01 Jan 1970 00:10:00 GMT'),1),600)
        self.assertEqual(retry_delay(failure(retry_after='3600'),100),3600)
        self.assertEqual(retry_delay(failure(retry_after='invalid'),100),300)
        self.assertEqual(retry_delay(failure(retry_after='NaN'),1),15)

    def test_transient_recovery(self):
        with patch('in1k10_transport.urllib.request.urlopen',side_effect=[failure(503,retry_after=None),ConnectionResetError('reset'),io.BytesIO(b'{}')]),patch('in1k10_transport.time.sleep') as sleep:
            self.assertEqual(API('key').call('/events'),{})
        self.assertEqual([call.args[0] for call in sleep.call_args_list],[1,2])

    def test_nontransient_error_not_retried(self):
        with patch('in1k10_transport.urllib.request.urlopen',side_effect=failure(403)) as request,patch('in1k10_transport.time.sleep') as sleep:
            with self.assertRaises(urllib.error.HTTPError):API('key').call('/events')
        self.assertEqual(request.call_count,1)
        sleep.assert_not_called()


if __name__=='__main__':unittest.main()
