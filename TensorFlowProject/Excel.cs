using Spire.Xls;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowProject
{
    public class Excel
    {
        private Workbook workbook = new Workbook();
        private Worksheet sheet;
        public Excel(bool isnew, string path = null)
        {
            if (isnew)
            {
                workbook = new Workbook();
                workbook.Worksheets[1].Remove();
            }
            else
                workbook.LoadFromFile(path);
            sheet = workbook.Worksheets[0];
        }
        public void Save(string path = null)
        {
            if (path == null) workbook.Save();
            else workbook.SaveToFile(path, ExcelVersion.Version2016);
            return;

        }
        public int GetRawCount() => sheet.Rows.Count();
        public int GetColumnCount() => sheet.Columns.Count();
        public string this[int row, int column]
        {
            get
            {
                string cln = ((char)(65 + column)).ToString();
                cln += row + 1;
                return sheet.Range[cln].Value;
            }
            set
            {
                string cln = ((char)(65 + column)).ToString();
                cln += row + 1;
                sheet.Range[cln].Value = value;
            }
        }
        public string this[string index]
        {
            get
            {
                return sheet.Range[index].Text;
            }
            set
            {
                sheet.Range[index].Text = value;
            }
        }


    }
}
