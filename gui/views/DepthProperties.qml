import QtQuick 2.0
import QtQuick.Layouts 1.3
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1

ListView {
    id: depthProperties
    delegate: Text {
        anchors.leftMargin: 50
        font.pointSize: 15
        horizontalAlignment: Text.AlignHCenter
        text: display
    }

    Rectangle {
        id: backgroundRect1
        color: "black"
        width: parent.width
        height: parent.height

        ComboBox {
            id: comboBox
            x: 0
            y: 102
            width: 195
            height: 33
            model: medianChoices
            onActivated: function(index) {
                depthBridge.setMedianFilter(model[index])
            }
            ToolTip.text: qsTr("Disparity / depth median filter kernel size (N x N) ")
            ToolTip.visible: hovered
        }

        Slider {
            id: dctSlider
            x: 360
            y: 89
            width: 200
            height: 25
            snapMode: RangeSlider.NoSnap
            stepSize: 1
            from: 0
            to: 255
            value: 240
            onValueChanged: {
                depthBridge.setDisparityConfidenceThreshold(value)
            }
            ToolTip.text: qsTr("Stereo depth algorithm searches for the matching feature from right camera point to the left image (along the 96 disparity levels). During this process it computes the cost for each disparity level and chooses the minimal cost between two disparities and uses it to compute the confidence at each pixel. Stereo node will output disparity/depth pixels only where depth confidence is below the confidence threshold (lower the confidence value means better depth accuracy).")
            ToolTip.visible: hovered
        }

        Text {
            id: text2
            x: 0
            y: 71
            width: 195
            height: 25
            color: "#ffffff"
            text: qsTr("Median filtering")
            font.pixelSize: 18
            font.styleName: "Regular"
            font.weight: Font.Medium
            font.family: "Microsoft YaHei"
        }

        Switch {
            id: switch1
            x: 0
            y: 187
            checked: lrc
            text: qsTr("<font color=\"white\">Left Right Check</font>")
            transformOrigin: Item.Center
            font.family: "Microsoft YaHei"
            autoExclusive: false
            onToggled: {
                depthBridge.toggleLeftRightCheck(switch1.checked)
            }
            ToolTip.text: qsTr("LR-Check is used to remove incorrectly calculated disparity pixels due to occlusions at object borders (Left and Right camera views are slightly different).")
            ToolTip.visible: hovered
        }

        Switch {
            id: switch5
            x: 275
            y: 0
            width: 167
            height: 38
            text: qsTr("<font color=\"white\">Enabled</font>")
            autoExclusive: false
            font.family: "Microsoft YaHei"
            checked: true
            transformOrigin: Item.Center
            onToggled: {
                appBridge.toggleDepth(switch5.checked)
            }
        }

        Switch {
            id: switch2
            x: 0
            y: 233
            text: qsTr("<font color=\"white\">Extended Disparity</font>")
            autoExclusive: false
            font.family: "Microsoft YaHei"
            transformOrigin: Item.Center
            onToggled: {
                depthBridge.toggleExtendedDisparity(switch2.checked)
            }
            ToolTip.text: qsTr("Extended disparity mode allows detecting closer distance objects for the given baseline. This increases the maximum disparity search from 96 to 191, meaning the range is now: [0..190]. So this cuts the minimum perceivable distance in half, given that the minimum distance is now focal_length * base_line_dist / 190 instead of focal_length * base_line_dist / 95.")
            ToolTip.visible: hovered
        }

        Switch {
            id: switch3
            x: 0
            y: 141
            text: qsTr("<font color=\"white\">Subpixel</font>")
            autoExclusive: false
            transformOrigin: Item.Center
            font.family: "Microsoft YaHei"
            onToggled: {
                depthBridge.toggleSubpixel(switch3.checked)
            }
            ToolTip.text: qsTr("Subpixel mode improves the precision and is especially useful for long range measurements. It also helps for better estimating surface normals.")
            ToolTip.visible: hovered
        }

        Text {
            id: text3
            x: 359
            y: 71
            width: 200
            height: 25
            color: "#ffffff"
            text: qsTr("Confidence Threshold")
            font.pixelSize: 18
            horizontalAlignment: Text.AlignHCenter
            font.styleName: "Regular"
            font.weight: Font.Medium
            font.family: "Microsoft YaHei"
        }

        Slider {
            id: sigmaSlider
            x: 362
            y: 138
            width: 200
            height: 25
            stepSize: 1
            snapMode: RangeSlider.NoSnap
            value: 0
            to: 255
            onValueChanged: {
                depthBridge.setBilateralSigma(value)
            }
            ToolTip.text: qsTr("Sigma value for Bilateral Filter applied on depth.")
            ToolTip.visible: hovered
        }

        Text {
            id: text5
            x: 566
            y: 95
            width: 17
            height: 25
            color: "#ffffff"
            text: dctSlider.value
            font.pixelSize: 12
        }

        Text {
            id: text6
            x: 360
            y: 120
            width: 200
            height: 25
            color: "#ffffff"
            text: qsTr("Bilateral Sigma")
            font.pixelSize: 18
            horizontalAlignment: Text.AlignHCenter
            font.styleName: "Regular"
            font.weight: Font.Medium
            font.family: "Microsoft YaHei"
        }

        Text {
            id: text8
            x: 566
            y: 136
            width: 17
            height: 20
            color: "#ffffff"
            text: sigmaSlider.value
            font.pixelSize: 12
            rotation: 0
        }

        Text {
            id: text9
            x: 362
            y: 169
            width: 200
            height: 25
            color: "#ffffff"
            text: qsTr("Depth Range [m]")
            font.pixelSize: 18
            horizontalAlignment: Text.AlignHCenter
            font.styleName: "Regular"
            font.weight: Font.Medium
            font.family: "Microsoft YaHei"
        }

        Text {
            id: text1
            x: 0
            y: 4
            width: 285
            height: 30
            color: "#ffffff"
            text: qsTr("Depth Properties")
            font.pixelSize: 26
            horizontalAlignment: Text.AlignHCenter
            font.styleName: "Regular"
            font.family: "Microsoft YaHei"
        }

        Text {
            id: text10
            x: 360
            y: 236
            width: 200
            height: 25
            color: "#ffffff"
            text: qsTr("LRC Threshold")
            font.pixelSize: 18
            horizontalAlignment: Text.AlignHCenter
            font.styleName: "Regular"
            font.weight: Font.Medium
            font.family: "Microsoft YaHei"
        }

        Slider {
            id: lrcSlider
            x: 361
            y: 256
            width: 198
            height: 27
            stepSize: 1
            to: 10
            value: 10
            from: 0
            onValueChanged: {
                depthBridge.setLrcThreshold(value)
            }
            ToolTip.text: qsTr("Disparity is considered for the output when the difference between LR and RL disparities is smaller than the LR check threshold.")
            ToolTip.visible: hovered
        }

        Text {
            id: text34
            x: 566
            y: 260
            width: 17
            height: 20
            color: "#ffffff"
            text: lrcSlider.value
            font.pixelSize: 12
            rotation: 0
        }

        Switch {
            id: switch6
            x: 400
            y: 0
            width: 200
            height: 38
            text: qsTr("<font color=\"white\">Use Disparity</font>")
            autoExclusive: false
            font.family: "Microsoft YaHei"
            transformOrigin: Item.Center
            onToggled: {
                appBridge.toggleDisparity(switch6.checked)
            }
        }

        Text {
            id: text32
            x: 343
            y: 197
            width: 40
            height: 25
            color: "#ffffff"
            text: qsTr("Min")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Microsoft YaHei"
        }

        TextField {
            id: textField6
            x: 381
            y: 197
            width: 60
            height: 25
            text: qsTr("0")
            placeholderText: "Min depth"
            bottomPadding: 5
            validator: DoubleValidator {
            }
            font.family: "Microsoft YaHei"
            onEditingFinished: {
                depthBridge.setDepthRange(textField6.text, textField7.text)
            }
        }

        TextField {
            id: textField7
            x: 499
            y: 197
            width: 60
            height: 25
            text: qsTr("10")
            placeholderText: "Max depth"
            bottomPadding: 5
            validator: DoubleValidator {
            }
            font.family: "Microsoft YaHei"
            onEditingFinished: {
                depthBridge.setDepthRange(textField6.text, textField7.text)
            }
        }

        Text {
            id: text33
            x: 456
            y: 197
            width: 40
            height: 25
            color: "#ffffff"
            text: qsTr("Max")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Microsoft YaHei"
        }

        Text {
            id: textirlaserdot
            x: 360
            y: 290
            width: 200
            height: 25
            color: "#ffffff"
            text: qsTr("IR Laser Dot Projector [mA]")
            font.pixelSize: 18
            horizontalAlignment: Text.AlignHCenter
            font.styleName: "Regular"
            font.weight: Font.Medium
            font.family: "Microsoft YaHei"
        }

        Slider {
            id: irlaserdotslider
            enabled: irEnabled
            x: 361
            y: 310
            width: 198
            height: 27
            stepSize: 1
            to: 1200
            value: irDotBrightness
            from: 0
            onValueChanged: {
                depthBridge.setIrLaserDotProjector(value)
            }
            ToolTip.text: qsTr("For OAK-D Pro: specify IR dot projector brightness.")
            ToolTip.visible: hovered
        }

        Text {
            id: irlaserdotslidervalue
            x: 566
            y: 315
            width: 17
            height: 20
            color: "#ffffff"
            text: irlaserdotslider.value
            font.pixelSize: 12
            rotation: 0
        }

        Text {
            id: textirfloodilluminator
            x: 360
            y: 350
            width: 200
            height: 25
            color: "#ffffff"
            text: qsTr("IR Flood Illuminator [mA]")
            font.pixelSize: 18
            horizontalAlignment: Text.AlignHCenter
            font.styleName: "Regular"
            font.weight: Font.Medium
            font.family: "Microsoft YaHei"
        }

        Slider {
            id: irfloodslider
            enabled: irEnabled
            x: 361
            y: 370
            width: 198
            height: 27
            stepSize: 1
            to: 1500
            value: irFloodBrightness
            from: 0
            onValueChanged: {
                depthBridge.setIrFloodIlluminator(value)
            }
            ToolTip.text: qsTr("For OAK-D Pro: specify IR flood illumination brightness.")
            ToolTip.visible: hovered
        }

        Text {
            id: irfloodslidervalue
            x: 566
            y: 375
            width: 17
            height: 20
            color: "#ffffff"
            text: irfloodslider.value
            font.pixelSize: 12
            rotation: 0
        }
    }
}
/*##^##
Designer {
    D{i:0;autoSize:true;height:480;width:640}
}
##^##*/
