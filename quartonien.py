import xlsxwriter
import numpy as np
# Create an new Excel file and add a worksheet.

workbook = xlsxwriter.Workbook('demo.xlsx')
worksheet = workbook.add_worksheet()

labels = np.random.rand(10,3)

worksheet.set_column('A:C', 10)
bold = workbook.add_format({'bold': True, 'center_across': True })
worksheet.write('A1', 'Distance', bold)
worksheet.write('B1', 'Tilt', bold)
worksheet.write('C1', 'Rotation', bold)
for idx, x in np.ndenumerate(labels):
    worksheet.write(idx[0]+1, idx[1], x)







"""

# Widen the first column to make the text clearer.
worksheet.set_column('A:A', 20)

# Add a bold format to use to highlight cells.
bold = workbook.add_format({'bold': True})

# Write some simple text.
worksheet.write('A1', 'Hello')

# Text with formatting.
worksheet.write('A2', 'World', bold)

# Write some numbers, with row/column notation.
worksheet.write(2, 0, 123)
worksheet.write(3, 0, 123.456)

"""


workbook.close()