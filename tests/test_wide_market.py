import unittest
import pandas as pd
from betalens_db_manager.wide_market import decimal4, normalize, b_share, NULL


class WideMarketTests(unittest.TestCase):
    def test_exact_precision_and_blank_column(self):
        self.assertEqual(decimal4(pd.Series(['', '999941.02', '1974786775.6515', '-0', '1E2'])).tolist(),
                         [NULL, '999941.0200', '1974786775.6515', '0.0000', '100.0000'])
        self.assertEqual(decimal4(pd.Series(['',''])).tolist(), [NULL,NULL])
        with self.assertRaises(ValueError):
            decimal4(pd.Series(['1.00001']))

    def test_sentinel_time_and_source_identity(self):
        frame=pd.DataFrame([{'代码':'000001.SZ','简称':'平安银行','日期':'1991-04-03',
            '前收盘价(元)':'40','开盘价(元)':'49','收盘价(元)':'49','市盈率':'3.40282346638529E38','Unnamed: 7':''}])
        result,audit=normalize(frame,list(frame.columns[:-1]),'source1')
        self.assertEqual(result['市盈率'].iloc[0],NULL)
        self.assertEqual(result['pe_suspect'].iloc[0],'1')
        self.assertEqual(audit[0]['original_value'],'3.40282346638529E38')
        self.assertEqual(result['open_time'].iloc[0],'1991-04-03 09:30:01')
        self.assertEqual(result['close_time'].iloc[0],'1991-04-03 15:00:01')
        self.assertEqual(result['source_row'].iloc[0],'1')
        self.assertEqual(result['代码'].iloc[0],'000001.SZ')
        self.assertNotIn('metric',result)

    def test_b_share_exclusion(self):
        for code in ['200002.SZ','900919.SH']:
            self.assertTrue(b_share(code))
        for code in ['000001.SZ','920305.BJ','801010.SI','159110.OF','600000.SH']:
            self.assertFalse(b_share(code))


if __name__=='__main__': unittest.main()
